"""
This module provides a mixin class for Nunchaku models. It is intended to be inherited by nunchaku models.
"""

from warnings import warn

import torch


class NunchakuModelMixin:
    """
    Mixin class for Nunchaku models.
    """

    offload: bool = False

    def set_offload(self, offload: bool, **kwargs):
        """
        Enable or disable CPU offloading for the model.

        Parameters
        ----------
        offload : bool
            If True, enable CPU offloading. If False, disable it.
        **kwargs
            Additional keyword arguments for custom offload logic.

        Raises
        ------
        NotImplementedError
            If not implemented in the subclass.
        """
        raise NotImplementedError("CPU offload needs to be implemented in the child class")

    def to_safely(self, *args, **kwargs):
        """
        Safely move the model to a device or change its dtype.

        This method overrides the default ``.to()`` behavior to:

        - Prevent moving the model to GPU if offload is enabled.
        - Prevent changing the dtype of quantized models.

        Parameters
        ----------
        *args
            Positional arguments for device or dtype.
        **kwargs
            Keyword arguments for device or dtype.

        Returns
        -------
        self : NunchakuModelMixin
            The model itself, possibly moved to a new device or dtype.

        Raises
        ------
        ValueError
            If attempting to cast a quantized model to a new dtype after initialization.

        Warns
        -----
        UserWarning
            If attempting to move the model to GPU while offload is enabled.
        """
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present_in_args = "dtype" in kwargs

        # Try converting arguments to torch.device in case they are passed as strings
        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if dtype_present_in_args:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please "
                "use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`"
            )
        if self.offload:
            if device_arg_or_kwarg_present:
                warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
                return self
        return self.to(*args, **kwargs)
