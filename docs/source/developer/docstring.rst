Writing Docstrings
==================

Use this guide to write clear, consistent Python docstrings for ComfyUI-nunchaku.
Follow the `NumPy style guide <https://numpydoc.readthedocs.io/en/latest/format.html>`__, and always specify variable shapes, dtypes, and notation.
The docstring should be concise and informative.

.. note::

   All docstrings will be automatically rendered in the :doc:`API documentation <../api/toc>` using Sphinx's autodoc extension.
   Well-written docstrings ensure the API reference is clear and helpful for users.

Docstring Format
----------------

A standard docstring should look like:

.. code-block:: python

    """
    Short summary of what the function or class does.

    (Optional) Extended description.

    Parameters
    ----------
    param1 : type
        Description.
    param2 : type, optional
        Description. Default is ...
    param3 : array-like, shape (n, m), dtype float
        Example of shape and dtype notation.

    Returns
    -------
    out1 : type
        Description.
    out2 : type
        Description.

    Raises
    ------
    ValueError
        When this exception is raised.

    See Also
    --------
    other_function : brief description

    Notes
    -----
    Additional details or references.

    Examples
    --------
    >>> result = func(1, 2)
    >>> print(result)
    3
    """

Guidelines
----------

- Use triple double quotes (``"""``) for all docstrings.
- Every public module, class, method, and function must have a docstring.
- The first line is a concise summary.
- Use sections in this order (as needed): ``Parameters``, ``Returns``, ``Raises``, ``See Also``, ``Notes``, ``Examples``.

Shapes, Dtypes, and Notation
-----------------------------

- Always specify expected shape and dtype for tensors/arrays.
- Use plain text for shapes (not LaTeX/math symbols).
- Use clear, single-letter or descriptive names for shape dimensions (e.g., `B` for batch size).
- Define all shape symbols in a ``Notes`` section.

**Example:**

.. code-block:: python

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W), dtype float32
            Input image tensor.
        mask : Optional[torch.Tensor], shape (B, 1, H, W), dtype bool
            Optional mask.

        Returns
        -------
        out : torch.Tensor, shape (B, num_classes), dtype float32
            Output logits.

        Raises
        ------
        ValueError
            If input shapes are incompatible.

        Notes
        -----
        Notations:
        - B: batch size
        - C: channels
        - H: height
        - W: width
        - num_classes: number of output classes

        Examples
        --------
        >>> x = torch.randn(8, 3, 224, 224)
        >>> out = model.forward(x)
        """
        ...

ComfyUI Node Docstrings
-----------------------

For ComfyUI nodes (classes that define ``INPUT_TYPES``, ``RETURN_TYPES``, etc.), follow these additional conventions:

Module-Level Docstring
~~~~~~~~~~~~~~~~~~~~~~

Start each node module with a brief description:

.. code-block:: python

    """
    This module provides the :class:`NunchakuFluxDiTLoader` class for loading Nunchaku FLUX models.
    It also supports attention implementation selection, CPU offload, and first-block caching.
    """

.. tip::

   Use ``:class:`ClassName``` to create cross-references to classes in the documentation.

Class-Level Docstring
~~~~~~~~~~~~~~~~~~~~~

Document the node's purpose and its persistent attributes:

.. code-block:: python

    class NunchakuFluxDiTLoader:
        """
        Loader for Nunchaku FLUX.1 models.

        This class manages model loading, device selection, attention implementation,
        CPU offload, and caching for efficient inference.

        Attributes
        ----------
        transformer : :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel` or None
            The loaded transformer model.
        metadata : dict or None
            Metadata associated with the loaded model.
        model_path : str or None
            Path to the loaded model.
        device : torch.device or None
            Device on which the model is loaded.
        """

Method Docstrings
~~~~~~~~~~~~~~~~~

For ``INPUT_TYPES`` (class method):

.. code-block:: python

    @classmethod
    def INPUT_TYPES(s):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """

For the main node function:

.. code-block:: python

    def apply(
        self,
        model,
        pulid_pipline: PuLIDPipeline,
        image,
        weight: float,
        start_at: float,
        end_at: float,
        attn_mask=None,
        options=None,
        unique_id=None,
    ):
        """
        Apply PuLID ID customization according to the given image to the model.

        Parameters
        ----------
        model : object
            The Nunchaku FLUX model to modify.
        pulid_pipline : :class:`~nunchaku.pipeline.pipeline_flux_pulid.PuLIDPipeline`
            The PuLID pipeline instance.
        image : np.ndarray or torch.Tensor
            The input image for identity embedding extraction.
        weight : float
            The strength of the identity guidance.
        start_at : float
            The starting step (as a fraction of total steps) to apply PuLID.
        end_at : float
            The ending step (as a fraction of total steps) to apply PuLID.
        attn_mask : torch.Tensor, optional
            Optional attention mask. Default is None.
        options : dict, optional
            Additional options. Default is None.
        unique_id : str, optional
            Unique identifier for the node. Default is None.

        Returns
        -------
        tuple
            A tuple containing the modified model.
        """

Best Practices
--------------

Writing Tips
~~~~~~~~~~~~

- **Be concise and clear.** Start with a short summary describing what the function or class does.
- **Document all parameters and return values.** Indicate if a parameter can be ``None`` or is optional.
- **Use cross-references.** Link to related classes using ``:class:`~module.ClassName``` for better navigation.
- **Include an** ``Examples`` **section** to demonstrate typical usage (especially for utility functions).
- **List all possible exceptions in a** ``Raises`` **section.**
- **Use a** ``Notes`` **section** to define shape symbols and explain special behaviors.
- **Add a** ``See Also`` **section** for related functions or methods.
- **Module docstrings matter.** Always include a module-level docstring explaining what the module provides.

Cross-Referencing
~~~~~~~~~~~~~~~~~

Use Sphinx roles to create links to other parts of the documentation:

- ``:class:`ClassName``` - Link to a class in the current module
- ``:class:`~full.module.path.ClassName``` - Link with short name (shows only ``ClassName``)
- ``:meth:`method_name``` - Link to a method
- ``:func:`function_name``` - Link to a function
- ``:doc:`../path/to/doc``` - Link to another documentation page

**Example:**

.. code-block:: python

    """
    See :class:`~nunchaku.models.transformers.transformer_flux.NunchakuFluxTransformer2dModel`
    for the underlying transformer implementation.

    Refer to :doc:`../get_started/installation` for setup instructions.
    """

Common Patterns
~~~~~~~~~~~~~~~

**For utility functions:**

.. code-block:: python

    def get_package_version(package_name: str) -> str:
        """
        Retrieve the version string for a given installed package.

        Parameters
        ----------
        package_name : str
            The name of the package to query.

        Returns
        -------
        str
            The version string, or an error message if not found.

        Examples
        --------
        >>> get_package_version("torch")
        '2.0.1'
        """

**For classes with initialization:**

.. code-block:: python

    def __init__(self):
        """
        Initialize the NunchakuFluxDiTLoader.

        Sets up internal state and selects the default torch device.
        """

Useful AI Prompts
~~~~~~~~~~~~~~~~~

When using AI assistants to help write or improve docstrings, consider these prompts:

.. code-block:: text

   Improve the writing of the docstring according to this guide. Be concise. Organize my comments clearly.

.. code-block:: text

   Write the docstring for this module (every class and functions) according to this guide.
   Each function and class should be shown properly and beautifully in sphinx html.
   Be concise. Organize my comments clearly. Use cross-references where appropriate.

.. code-block:: text

   Add a module-level docstring following the NumPy style guide. Mention the main classes
   and functions with cross-references.

Testing Your Docstrings
~~~~~~~~~~~~~~~~~~~~~~~~

After writing docstrings, verify they render correctly:

1. Build the documentation locally (see :doc:`build_docs`)
2. Check the generated API documentation under ``docs/build/html/api/``
3. Ensure cross-references work and formatting is correct

For further questions or formatting help, refer to existing ComfyUI-nunchaku code or ask in the developer chat.
