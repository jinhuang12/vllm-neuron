# SPDX-License-Identifier: Apache-2.0
import operator
import logging
import torch

from .base import FXPass


logger = logging.getLogger(__name__)


# View ops that preserve element order over a contiguous base, so a write
# through the view can be materialized back to the base with a single
# reshape of the out-of-place result.  Non-reshape views (transpose,
# permute, slice, select, ...) would need a scatter-back instead and are
# rejected loudly in _rewire_base_readers.
_RESHAPE_ONLY_VIEWS = {
    "view",
    "reshape",
    "flatten",
    "unflatten",
    "squeeze",
    "unsqueeze",
    "view_as",
    "reshape_as",
    "_unsafe_view",
}


class InPlaceToOutOfPlacePass(FXPass):
    """Convert in-place operations to out-of-place equivalents.

    For example, ``x.add_(y)`` becomes ``x_modified = x.add(y)`` and all
    subsequent references to ``x`` are rewritten to use ``x_modified``.
    Runs after the aliasing pass and is required because the XLA/HLO
    backend does not support in-place semantics.
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "inplace_to_outofplace"

    def run(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> tuple[torch.fx.GraphModule, dict]:
        """Rewrite all in-place method calls to their out-of-place equivalents.

        Args:
            gm: The PyTorch FX GraphModule to transform.
            **kwargs: Additional arguments passed from the pass manager.

        Returns:
            Tuple of (transformed GraphModule, conversion-count metadata dict
            with keys ``inplace_methods``, ``copy_``, ``setitem`` and
            ``base_reader_rewires``).

        Raises:
            RuntimeError: If the conversion fails.
        """
        try:
            counts = self._convert_inplace_ops(gm)
            gm.recompile()
            return gm, counts
        except Exception as e:
            raise RuntimeError(f"Inplace to outofplace conversion failed: {e}") from e

    # =========================================================================
    # In-place operation conversion
    # =========================================================================

    def _get_outofplace_equivalent(self, inplace_op: str) -> str | None:
        """Return the out-of-place name for an in-place op.

        For example, ``'add_'`` → ``'add'``.

        Args:
            inplace_op: The in-place operation name (e.g. ``'add_'``).

        Returns:
            The out-of-place equivalent name, or None if no equivalent
            exists on ``torch.Tensor``.
        """
        outofplace_op = inplace_op[:-1]
        return outofplace_op if hasattr(torch.Tensor, outofplace_op) else None

    def _convert_inplace_ops(self, gm: torch.fx.GraphModule) -> dict:
        """Convert in-place method calls on graph inputs to out-of-place equivalents.

        Args:
            gm: The FX GraphModule whose graph will be mutated in place.

        Returns:
            Conversion counts, keyed by conversion kind.
        """
        counts = {
            "inplace_methods": 0,
            "copy_": 0,
            "setitem": 0,
            "base_reader_rewires": 0,
        }
        # Snapshot the node list: we mutate the graph (erase/insert) during
        # iteration, so iterating over a live view would skip or revisit nodes.
        for node in list(gm.graph.nodes):
            if node.op == "call_function" and node.target == operator.setitem:
                self._convert_setitem(gm, node, counts)
                continue

            # Only target single-trailing-underscore methods (e.g. add_, copy_).
            # Double-underscore dunder methods (__setitem__) are not in-place ops.
            if node.op != "call_method" or not (
                node.target.endswith("_") and not node.target.startswith("__")
            ):
                continue

            # args[0] is always the tensor being mutated (self in method call).
            original_input = node.args[0]

            # root_input is the base placeholder stashed by the aliasing pass
            # when the mutation traces back through views.  Read it up front:
            # the mutated view and the base need separate rewiring below.
            root_input = node.meta.get("root_input")

            if node.target == "copy_":
                source = node.args[1]
                # copy_ is in-place and XLA/HLO cannot lower it.
                # Replace with expand_as + slice_scatter which produces
                # a new tensor.  expand_as is needed because copy_
                # accepts a broadcastable source, so we must match the
                # source shape to the dest shape before slice_scatter.
                with gm.graph.inserting_before(node):
                    expanded = gm.graph.call_method(
                        "expand_as", args=(source, original_input)
                    )
                    slice_scatter = gm.graph.call_function(
                        torch.slice_scatter,
                        args=(original_input, expanded),
                    )
                if hasattr(original_input, "name"):
                    slice_scatter.name = f"{original_input.name}_modified"
                node.replace_all_uses_with(slice_scatter)
                gm.graph.erase_node(node)
                self._update_subsequent_ops(gm, slice_scatter, original_input)
                counts["copy_"] += 1
                counts["base_reader_rewires"] += self._rewire_base_readers(
                    gm, slice_scatter, original_input, root_input
                )
                continue
            else:
                # General case: swap the in-place target for its out-of-place
                # equivalent (e.g. add_ → add).  The node now produces a new
                # tensor instead of mutating the original.
                outofplace_op = self._get_outofplace_equivalent(node.target)
                if not outofplace_op:
                    raise NotImplementedError(f"'{node.target}' is not supported")
                node.target = outofplace_op
                counts["inplace_methods"] += 1

            # Rename the node so later passes (and debug output) can tell
            # which input was modified.  root_input is set by the aliasing
            # pass when the mutation traces back through views.
            if hasattr(original_input, "name"):
                node.name = (
                    f"{node.meta.get('root_input', original_input.name)}_modified"
                )

            # Rewrite all downstream references from the original input to
            # the new out-of-place result so the SSA form stays valid.
            self._update_subsequent_ops(gm, node, original_input)

            # When the mutation went through a view chain, later readers that
            # re-derive from the BASE tensor still bind the pre-mutation
            # value; rewire them to a base-shaped updated tensor.
            counts["base_reader_rewires"] += self._rewire_base_readers(
                gm, node, original_input, root_input
            )

        return counts

    # =========================================================================
    # Base-reader rewiring for view-mediated mutations
    # =========================================================================

    def _is_reshape_only_view_chain(
        self, view_node: torch.fx.Node, root_input: torch.fx.Node
    ) -> bool:
        """Return True if *view_node* derives from *root_input* purely through
        element-order-preserving reshape-class views.

        Args:
            view_node: The view node the in-place op mutated.
            root_input: The base placeholder the chain must terminate at.

        Returns:
            True if every hop from *view_node* back to *root_input* is in
            ``_RESHAPE_ONLY_VIEWS``, False otherwise.
        """
        current = view_node
        visited: set[torch.fx.Node] = set()
        while isinstance(current, torch.fx.Node) and current not in visited:
            if current is root_input:
                return True
            visited.add(current)

            if current.op == "call_method" and isinstance(current.target, str):
                op_name = current.target
            elif current.op == "call_function":
                # ATen targets may carry an overload suffix (e.g.
                # "view.default") — strip it before checking the set.
                op_name = getattr(current.target, "__name__", str(current.target))
                op_name = op_name.split(".")[0] if "." in op_name else op_name
            else:
                return False

            if op_name not in _RESHAPE_ONLY_VIEWS:
                return False
            if not current.args or not isinstance(current.args[0], torch.fx.Node):
                return False
            current = current.args[0]
        return False

    def _rewire_base_readers(
        self,
        gm: torch.fx.GraphModule,
        result_node: torch.fx.Node,
        mutated_input,
        root_input,
    ) -> int:
        """Rewire post-mutation readers of the BASE tensor after converting a
        view-mediated in-place op to its out-of-place form.

        For ``flat = base.view(...); flat.index_put_(dest, vals)`` the
        conversion produces a view-shaped out-of-place result whose later
        VIEW uses are rewritten by ``_update_subsequent_ops`` — but a reader
        that re-derives from the BASE (``base.reshape(...)``) still binds the
        pre-mutation tensor, freezing a read-before-write into the lowered
        HLO.  Materialize a base-shaped updated tensor from the result and
        rewrite subsequent uses of the base to it (dominance-safe: only nodes
        after the insertion point are rewritten).

        Args:
            gm: The FX GraphModule whose graph will be mutated.
            result_node: The out-of-place result of the converted mutation.
            mutated_input: The tensor the in-place op targeted (``args[0]``).
            root_input: The base placeholder traced by the aliasing pass
                (``node.meta['root_input']``), or None.

        Returns:
            1 if a base-shaped update was materialized and rewired, else 0.

        Raises:
            NotImplementedError: If the view chain is not reshape-only (a
                scatter-back would be required).
            RuntimeError: If the base placeholder carries no shape metadata.
        """
        if not isinstance(root_input, torch.fx.Node) or root_input is mutated_input:
            return 0
        if root_input.op != "placeholder":
            return 0

        # Nothing to fix if no node after the mutation still reads the base.
        nodes_list = list(gm.graph.nodes)
        start_idx = nodes_list.index(result_node) + 1
        if not any(
            root_input in later.all_input_nodes for later in nodes_list[start_idx:]
        ):
            return 0

        if not self._is_reshape_only_view_chain(mutated_input, root_input):
            raise NotImplementedError(
                f"'{result_node.name}' mutates '{root_input.name}' through a "
                f"non-reshape view chain; materializing the update would need "
                f"a scatter-back, which is not implemented"
            )

        base_value = root_input.meta.get("example_value")
        if base_value is None or not hasattr(base_value, "shape"):
            raise RuntimeError(
                f"cannot materialize base-shaped update for "
                f"'{root_input.name}': placeholder has no example_value "
                f"shape metadata"
            )

        with gm.graph.inserting_after(result_node):
            updated_base = gm.graph.call_method(
                "reshape", args=(result_node, list(base_value.shape))
            )
        updated_base.meta["example_value"] = base_value
        updated_base.name = f"{root_input.name}_updated"

        # Rewrite all downstream references from the base placeholder to the
        # materialized post-mutation value so no reader sees the pre-write
        # tensor.
        self._update_subsequent_ops(gm, updated_base, root_input)
        return 1

    # =========================================================================
    # setitem conversion
    # =========================================================================

    def _convert_setitem(
        self, gm: torch.fx.GraphModule, node: torch.fx.Node, counts: dict
    ) -> None:
        """Convert ``operator.setitem(buf, idx, value)`` to scatter ops.

        Args:
            gm: The FX GraphModule whose graph will be mutated in place.
            node: The setitem node to convert.
            counts: Conversion-count dict updated in place.
        """
        buf, idx, value = node.args

        # Read the aliasing pass's base placeholder before the node is erased.
        root_input = node.meta.get("root_input")

        with gm.graph.inserting_before(node):
            scatter_node = self._build_scatter(gm, buf, idx, value)

        if scatter_node is None:
            # Unsupported — just rewrite subsequent refs and keep the setitem.
            if hasattr(buf, "name"):
                node.name = f"{node.meta.get('root_input', buf.name)}_modified"
            self._update_subsequent_ops(gm, node, buf)
            return

        if hasattr(buf, "name"):
            scatter_node.name = f"{node.meta.get('root_input', buf.name)}_modified"
        node.replace_all_uses_with(scatter_node)
        gm.graph.erase_node(node)
        self._update_subsequent_ops(gm, scatter_node, buf)
        counts["setitem"] += 1

        # When the write went through a view chain, rewire post-write readers
        # of the BASE tensor to a base-shaped updated value.
        counts["base_reader_rewires"] += self._rewire_base_readers(
            gm, scatter_node, buf, root_input
        )

    def _ensure_tensor_node(
        self, gm: torch.fx.GraphModule, value, buf
    ) -> torch.fx.Node:
        """Wrap a scalar/constant *value* into a ``full_like`` graph node.

        If *value* is already an FX Node it is returned unchanged.

        Args:
            gm: The FX GraphModule to insert nodes into.
            value: A scalar constant or an existing FX Node.
            buf: The reference tensor node used for ``full_like``.

        Returns:
            An FX Node representing the tensor value.
        """
        if isinstance(value, torch.fx.Node):
            return value
        return gm.graph.call_function(
            torch.full_like, args=(buf,), kwargs={"fill_value": value}
        )

    def _ensure_select_src(
        self, gm: torch.fx.GraphModule, value, buf, dim: int, idx: int
    ) -> torch.fx.Node:
        """Ensure *value* is a tensor shaped for ``select_scatter(buf, src, dim, idx)``.

        ``select_scatter`` expects *src* to have the shape of *buf* with
        dimension *dim* removed.  When *value* is a scalar we use
        ``torch.select`` to obtain a correctly-shaped reference tensor,
        then ``full_like`` to broadcast the scalar into that shape.

        Args:
            gm: The FX GraphModule to insert nodes into.
            value: A scalar constant or an existing FX Node.
            buf: The buffer tensor node.
            dim: The dimension to select along.
            idx: The index within *dim*.

        Returns:
            An FX Node with the correct shape for ``select_scatter``.
        """
        if isinstance(value, torch.fx.Node):
            return value
        ref = gm.graph.call_function(torch.select, args=(buf, dim, idx))
        return gm.graph.call_function(
            torch.full_like, args=(ref,), kwargs={"fill_value": value}
        )

    def _build_scatter(
        self, gm: torch.fx.GraphModule, buf, idx, value
    ) -> torch.fx.Node:
        """Build the appropriate scatter node for a given index type.

        Args:
            gm: The FX GraphModule to insert nodes into.
            buf: The buffer tensor node being scattered into.
            idx: The index (int, slice, tuple, or FX Node).
            value: The value to scatter.

        Returns:
            An FX Node representing the scatter result, or None if the
            index type is unsupported.

        Raises:
            NotImplementedError: If *idx* is an unsupported type.
        """
        if isinstance(idx, int):
            value = self._ensure_select_src(gm, value, buf, 0, idx)
            return gm.graph.call_function(
                torch.select_scatter, args=(buf, value, 0, idx)
            )
        if isinstance(idx, slice):
            return self._slice_scatter(gm, buf, value, idx, dim=0)
        if isinstance(idx, tuple):
            return self._tuple_scatter(gm, buf, value, idx)
        # Tensor index (or FX Node representing a tensor).
        # Bool masks: use torch.where(mask, value, buf) to avoid
        # index_put with bool indices which XLA can't lower.
        # Integer tensors: use index_put.
        if isinstance(idx, torch.fx.Node):
            value = self._ensure_tensor_node(gm, value, buf)
            ev = idx.meta.get("example_value", None)
            if ev is not None and ev.dtype == torch.bool:
                return gm.graph.call_function(torch.where, args=(idx, value, buf))
            return gm.graph.call_method("index_put", args=(buf, (idx,), value))
        raise NotImplementedError(f"setitem index type {type(idx)} not supported")

    def _slice_scatter(
        self, gm: torch.fx.GraphModule, buf, value, s: slice, dim: int
    ) -> torch.fx.Node:
        """Emit a scatter for a single slice on *dim*.

        When start == 0 and step == 1, uses ``torch.slice_scatter``.
        Otherwise, uses pad+select lowering like XLA.

        Args:
            gm: The FX GraphModule to insert nodes into.
            buf: The buffer tensor node being scattered into.
            value: The value to scatter into the slice region.
            s: The slice object describing the region.
            dim: The dimension to scatter along.

        Returns:
            An FX Node representing the scatter result.
        """
        start = s.start if s.start is not None else 0
        step = s.step if s.step is not None else 1

        # When value is a scalar, we need a tensor with the shape of the
        # sliced region.  Slice buf first to get the right shape reference,
        # then broadcast the scalar into that shape via full_like.
        if not isinstance(value, torch.fx.Node):
            region = gm.graph.call_function(
                torch.ops.aten.slice.Tensor,
                args=(buf,),
                kwargs=self._slice_kwargs(s, dim),
            )
            value = self._ensure_tensor_node(gm, value, region)

        if start == 0:
            kwargs: dict = {"dim": dim}
            if s.stop is not None:
                kwargs["end"] = s.stop
            if step > 1:
                kwargs["step"] = step
            return gm.graph.call_function(
                torch.slice_scatter, args=(buf, value), kwargs=kwargs
            )

        # Non-zero start with step > 1 or negative start: fall back
        # to slice_scatter directly (XLA handles small offsets fine;
        # the pad+where path only helps for large offsets).
        if step != 1 or start < 0:
            kwargs: dict = {"dim": dim}
            if s.start is not None:
                kwargs["start"] = s.start
            if s.stop is not None:
                kwargs["end"] = s.stop
            if s.step is not None:
                kwargs["step"] = s.step
            return gm.graph.call_function(
                torch.slice_scatter, args=(buf, value), kwargs=kwargs
            )

        # Non-zero positive start, step == 1: pad value and mask to
        # buf shape and use torch.where.  This mirrors the HLO
        # pattern (pad + select) that XLA produces for setitem.
        ev = buf.meta.get("example_value", None)
        if ev is None:
            # No shape metadata — fall back to slice_scatter.
            return gm.graph.call_function(
                torch.slice_scatter,
                args=(buf, value),
                kwargs={"dim": dim, "start": start, "end": s.stop, "step": step},
            )

        ndim = len(ev.shape)
        buf_size = ev.shape
        pos_dim = dim if dim >= 0 else ndim + dim

        stop = s.stop if s.stop is not None else buf_size[pos_dim]
        right_pad = buf_size[pos_dim] - stop

        # Expand value to match buf's rank so pad is a single multi-dim op.
        region = gm.graph.call_function(
            torch.ops.aten.slice.Tensor,
            args=(buf,),
            kwargs=self._slice_kwargs(s, dim),
        )
        value = gm.graph.call_method("expand_as", args=(value, region))

        # F.pad expects pairs in reverse dim order: [last_left, last_right,
        # ..., pos_dim_left, pos_dim_right].  We only need entries from the
        # last dim back to pos_dim; earlier dims are unaffected.
        pad_args = []
        for d in range(ndim - 1, pos_dim - 1, -1):
            if d == pos_dim:
                pad_args.extend([start, right_pad])
            else:
                pad_args.extend([0, 0])

        # Pad value to buf shape, then use a boolean mask of the same shape
        # to select between the padded value and the original buf.  This
        # mirrors the HLO pad+select pattern that XLA produces for setitem.
        padded_value = gm.graph.call_function(
            torch.nn.functional.pad,
            args=(value, pad_args),
            kwargs={"value": 0},
        )
        mask = gm.graph.call_function(
            torch.ones_like,
            args=(value,),
            kwargs={"dtype": torch.bool},
        )
        padded_mask = gm.graph.call_function(
            torch.nn.functional.pad,
            args=(mask, pad_args),
            kwargs={"value": False},
        )
        return gm.graph.call_function(
            torch.where, args=(padded_mask, padded_value, buf)
        )

    @staticmethod
    def _resolve_tuple_index(idx: tuple) -> list:
        """Expand Ellipsis and assign dim indices to tuple elements.

        Elements before Ellipsis get positive dims (0, 1, ...).
        Elements after Ellipsis get negative dims (-N, ..., -1).
        Trivial ``slice(None)`` entries are dropped.

        Args:
            idx: The tuple index, possibly containing Ellipsis.

        Returns:
            A list of ``(dim, index_element)`` pairs.
        """
        if Ellipsis not in idx:
            return [
                (dim, s)
                for dim, s in enumerate(idx)
                if not (isinstance(s, slice) and s == slice(None))
            ]

        ellipsis_pos = idx.index(Ellipsis)
        before = idx[:ellipsis_pos]
        after = idx[ellipsis_pos + 1 :]

        result = []
        for i, s in enumerate(before):
            if not (isinstance(s, slice) and s == slice(None)):
                result.append((i, s))
        for i, s in enumerate(after):
            dim = -(len(after) - i)
            if not (isinstance(s, slice) and s == slice(None)):
                result.append((dim, s))
        return result

    def _tuple_scatter(
        self, gm: torch.fx.GraphModule, buf, value, idx: tuple
    ) -> torch.fx.Node:
        """Handle tuple index by chaining ``slice_scatter`` from inner to outer.

        Single non-trivial dim → one ``slice_scatter``.
        Multiple non-trivial dims → slice *buf* down to inner dims,
        scatter *value* in, then scatter back up.

        Supports Ellipsis: elements before it get positive dims,
        elements after it get negative dims (counting from the end).

        Args:
            gm: The FX GraphModule to insert nodes into.
            buf: The buffer tensor node being scattered into.
            value: The value to scatter.
            idx: The tuple index.

        Returns:
            An FX Node representing the scatter result.
        """
        nontrivial = self._resolve_tuple_index(idx)

        # Tuple contains an FX Node (tensor/bool mask) -- fall back to index_put
        if any(isinstance(s, torch.fx.Node) for _, s in nontrivial):
            indices = tuple(s for _, s in nontrivial if isinstance(s, torch.fx.Node))
            return gm.graph.call_method("index_put", args=(buf, indices, value))

        if not nontrivial:
            return gm.graph.call_function(
                torch.slice_scatter, args=(buf, value), kwargs={"dim": 0}
            )

        if len(nontrivial) == 1:
            dim, s = nontrivial[0]
            if isinstance(s, int):
                value = self._ensure_select_src(gm, value, buf, dim, s)
                return gm.graph.call_function(
                    torch.select_scatter, args=(buf, value, dim, s)
                )
            return self._slice_scatter(gm, buf, value, s, dim)

        # Multiple non-trivial dims require a two-phase approach:
        #   Forward:  slice buf down through each outer dim to isolate the
        #             innermost region.
        #   Reverse:  scatter the modified inner region back up through each
        #             outer dim, reconstructing the full buffer.
        # This chaining lowers to HLO dynamic-update-slice correctly.
        outer_dims = nontrivial[:-1]
        inner_dim, inner_s = nontrivial[-1]

        # Keep a reference to each intermediate sliced buffer so the
        # reverse pass can scatter into the correct parent level.
        sliced_bufs = [buf]
        current = buf
        for dim, s in outer_dims:
            kwargs = self._slice_kwargs(s, dim)
            current = gm.graph.call_function(
                torch.ops.aten.slice.Tensor, args=(current,), kwargs=kwargs
            )
            sliced_bufs.append(current)

        if isinstance(inner_s, int):
            value = self._ensure_select_src(gm, value, current, inner_dim, inner_s)
            current = gm.graph.call_function(
                torch.select_scatter, args=(current, value, inner_dim, inner_s)
            )
        else:
            current = self._slice_scatter(gm, current, value, inner_s, inner_dim)

        # Reverse: scatter back up, using the sliced buffer at each level
        # as the input (not the original full buf).
        for i, (dim, s) in enumerate(reversed(outer_dims)):
            parent = sliced_bufs[len(outer_dims) - 1 - i]
            kwargs = self._slice_kwargs(s, dim)
            current = gm.graph.call_function(
                torch.slice_scatter, args=(parent, current), kwargs=kwargs
            )

        return current

    @staticmethod
    def _slice_kwargs(s, dim: int) -> dict:
        """Build kwargs dict for ``aten.slice.Tensor`` or ``slice_scatter``.

        Args:
            s: A slice object or int index.
            dim: The dimension the slice applies to.

        Returns:
            A kwargs dict suitable for ``aten.slice.Tensor`` or
            ``torch.slice_scatter``.
        """
        kwargs: dict = {"dim": dim}
        if isinstance(s, int):
            kwargs["start"] = s
            kwargs["end"] = s + 1
        else:
            if s.start is not None:
                kwargs["start"] = s.start
            if s.stop is not None:
                kwargs["end"] = s.stop
            if s.step is not None:
                kwargs["step"] = s.step
        return kwargs

    def _update_subsequent_ops(
        self,
        gm: torch.fx.GraphModule,
        modified_node: torch.fx.Node,
        original_input: torch.fx.Node,
    ) -> None:
        """Rewrite later nodes that reference *original_input* to use
        *modified_node* instead.

        Args:
            gm: The FX GraphModule whose graph will be mutated.
            modified_node: The replacement node.
            original_input: The node whose references should be replaced.
        """
        nodes_list = list(gm.graph.nodes)
        # Only rewrite nodes that appear *after* modified_node in the graph
        # to preserve SSA dominance: every use must be dominated by its def.
        start_idx = nodes_list.index(modified_node) + 1

        for later_node in nodes_list[start_idx:]:
            if later_node.op in ("call_method", "call_function", "output"):
                later_node.args = self._replace_in_structure(
                    later_node.args, original_input, modified_node
                )
                later_node.kwargs = self._replace_in_structure(
                    later_node.kwargs, original_input, modified_node
                )

    def _replace_in_structure(self, structure, old_node, new_node):
        """Recursively replace *old_node* with *new_node* in a nested structure.

        Args:
            structure: A nested combination of tuples, lists, and dicts.
            old_node: The node to find and replace.
            new_node: The replacement node.

        Returns:
            The structure with all occurrences of *old_node* replaced.
        """
        if structure is old_node:
            return new_node
        if isinstance(structure, tuple):
            return tuple(
                self._replace_in_structure(item, old_node, new_node)
                for item in structure
            )
        if isinstance(structure, list):
            return [
                self._replace_in_structure(item, old_node, new_node)
                for item in structure
            ]
        if isinstance(structure, dict):
            return {
                k: self._replace_in_structure(v, old_node, new_node)
                for k, v in structure.items()
            }
        return structure
