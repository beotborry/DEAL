import warnings
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from torchio.data.subject import Subject
from torchio.transforms.preprocessing.intensity.normalization_transform import NormalizationTransform
from torchio.transforms.preprocessing.intensity.normalization_transform import TypeMaskingMethod

TypeDoubleFloat = Tuple[float, float]


class RescaleIntensity(NormalizationTransform):
    """Rescale intensity values to a certain range.

    Args:
        out_min_max: Range :math:`(n_{min}, n_{max})` of output intensities.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (-d, d)`.
        percentiles: Percentile values of the input image that will be mapped
            to :math:`(n_{min}, n_{max})`. They can be used for contrast
            stretching, as in `this scikit-image example`_. For example,
            Isensee et al. use ``(0.5, 99.5)`` in their `nn-UNet paper`_.
            If only one value :math:`d` is provided,
            :math:`(n_{min}, n_{max}) = (0, d)`.
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        in_min_max: Range :math:`(m_{min}, m_{max})` of input intensities that
            will be mapped to :math:`(n_{min}, n_{max})`. If ``None``, the
            minimum and maximum input intensities will be used.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> ct = tio.ScalarImage('ct_scan.nii.gz')
        >>> ct_air, ct_bone = -1000, 1000
        >>> rescale = tio.RescaleIntensity(
        ...     out_min_max=(-1, 1), in_min_max=(ct_air, ct_bone))
        >>> ct_normalized = rescale(ct)

    .. _this scikit-image example: https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    .. _nn-UNet paper: https://arxiv.org/abs/1809.10486
    """  # noqa: B950

    def __init__(
        self,
        out_min_max: TypeDoubleFloat = (0, 1),
        percentiles: TypeDoubleFloat = (0, 100),
        masking_method: TypeMaskingMethod = None,
        in_min_max: Optional[TypeDoubleFloat] = None,
        **kwargs,
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.out_min_max = out_min_max
        self.in_min_max = in_min_max
        self.out_min, self.out_max = self._parse_range(
            out_min_max,
            'out_min_max',
        )
        self.percentiles = self._parse_range(
            percentiles,
            'percentiles',
            min_constraint=0,
            max_constraint=100,
        )

        if self.in_min_max is not None:
            self.in_min_max = self._parse_range(
                self.in_min_max,
                'in_min_max',
            )

        self.args_names = [
            'out_min_max',
            'percentiles',
            'masking_method',
            'in_min_max',
        ]

    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        image.set_data(self.rescale(image.data, mask, image_name))

    def rescale(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        image_name: str,
    ) -> torch.Tensor:
        # The tensor is cloned as in-place operations will be used
        array = tensor.clone().float().numpy()
        mask = mask.numpy()
        if not mask.any():
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because the mask to compute the statistics is empty'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor

        mask = array != 0.
        values = array[mask]
        cutoff = np.percentile(values, self.percentiles)
        np.clip(array, *cutoff, out=array)  # type: ignore[call-overload]

        if self.in_min_max is None:
            in_min, in_max = array.min(), array.max()
        else:
            in_min, in_max = self.in_min_max
        in_range = in_max - in_min
        if in_range == 0:  # should this be compared using a tolerance?
            message = (
                f'Rescaling image "{image_name}" not possible'
                ' because all the intensity values are the same'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return tensor

        out_range = self.out_max - self.out_min

        array -= in_min
        array /= in_range
        array *= out_range
        array += self.out_min
        return torch.as_tensor(array)


class ZNormalization(NormalizationTransform):
    """Subtract mean and divide by standard deviation.

    Args:
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, exclude_bg=False, masking_method: TypeMaskingMethod = None, **kwargs):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ['masking_method']
        self.exclude_bg = exclude_bg

    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        standardized = self.znorm(
            image.data,
            mask,
            exclude_bg = self.exclude_bg
        )
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image.path})'
            )
            raise RuntimeError(message)
        image.set_data(standardized)

    @staticmethod
    def znorm(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        exclude_bg: bool = True
    ) -> Optional[torch.Tensor]:
        tensor = tensor.clone().float()

        if exclude_bg:
            mask = tensor != 0.
            values = tensor[mask]
            mean, std = values.mean(), values.std()
            if std == 0:
                return None
            tensor -= mean
            tensor /= std
            return tensor
        else:
            mean, std = tensor.mean(), tensor.std()
            if std == 0:
                return None
            tensor -= mean
            tensor /= std
            return tensor
