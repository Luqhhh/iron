"""Auditable first-order DNNR arithmetic; no data loading or fit scheduling.

Mechanism source: Nader, Sixt and Landgraf, ICML 2022, and author package
ca6070734a659a51cbbdfaf9b97e508e8843a1bd. This independent implementation
uses an explicit centered-distance derivative, avoiding the author's dense
centering construction. See docs/dnnr_taylor/SOURCE_AUDIT.md for scope.
"""
import numpy as np


def _array(value,ndim):
    value=np.asarray(value,dtype=np.float64)
    if value.ndim!=ndim or not value.size or not np.isfinite(value).all():
        raise ValueError('Nonempty finite float64 array required')
    return value


def first_order_coefficients(anchor_x,anchor_y,neighbor_x,neighbor_y):
    """Uniform least-squares derivative anchored at a training point's label."""
    center=_array(anchor_x,1);points=_array(neighbor_x,2);targets=_array(neighbor_y,1)
    if points.shape!=(len(targets),len(center)) or not np.isfinite(anchor_y):
        raise ValueError('Taylor training-neighborhood shape/value mismatch')
    coefficient=np.linalg.lstsq(points-center,targets-float(anchor_y),rcond=None)[0]
    if not np.isfinite(coefficient).all():raise ValueError('Nonfinite local derivative')
    return coefficient


def first_order_prediction(query_x,anchor_x,anchor_y,coefficient):
    query=_array(query_x,1);center=_array(anchor_x,1);gradient=_array(coefficient,1)
    if query.shape!=center.shape or gradient.shape!=center.shape or not np.isfinite(anchor_y):
        raise ValueError('Taylor query/anchor/derivative shape mismatch')
    return float(anchor_y)+float(gradient@(query-center))


def scaling_cost_gradient(scale,delta,absolute_linear_errors,epsilon=1e-6):
    """Negative centered cosine similarity and its diagonal-scale gradient.

    Neighbor identities and local absolute linearization errors are held fixed
    for this derivative. Changes to the neighbor graph are discrete and are
    outside this numerical term. No validation/query target enters this API.
    """
    scale=_array(scale,1);delta=_array(delta,2);errors=_array(absolute_linear_errors,1)
    if (delta.shape!=(len(errors),len(scale)) or (errors<0).any()
            or not np.isfinite(epsilon) or epsilon<=0):
        raise ValueError('Scaling neighborhood/epsilon mismatch')
    raw_distances=np.linalg.norm(delta*scale,axis=1)
    distances=np.maximum(raw_distances,epsilon)
    a=errors-errors.mean();b=distances-distances.mean()
    # Preserve the author method's degenerate-neighborhood convention.
    if np.allclose(a,0) or np.allclose(b,0):return 0.,np.zeros_like(scale)
    norm_a=np.linalg.norm(a);norm_b=np.linalg.norm(b)
    cosine=float(a@b/(norm_a*norm_b))
    derivative_b=-a/(norm_a*norm_b)+cosine*b/(norm_b*norm_b)
    derivative_h=derivative_b-derivative_b.mean()
    # The distance floor is constant below epsilon, so its derivative is zero.
    derivative_h=derivative_h*(raw_distances>epsilon)
    gradient=np.sum((derivative_h/distances)[:,None]*delta**2*scale,axis=0)
    if not np.isfinite(gradient).all():raise ValueError('Nonfinite scaling derivative')
    return -cosine,gradient


def local_linear_errors(anchor_x,anchor_y,neighbor_x,neighbor_y):
    """Author scaling objective's local fit in unscaled feature coordinates.

    The author uses a pseudoinverse of the normal matrix in the scaling loss;
    its normal-matrix arithmetic is preserved here for source comparison.
    Prediction derivatives above use direct least squares, as its default
    prediction solver does. Both functions consume supplied training rows only.
    """
    center=_array(anchor_x,1);points=_array(neighbor_x,2);targets=_array(neighbor_y,1)
    if points.shape!=(len(targets),len(center)) or not np.isfinite(anchor_y):
        raise ValueError('Scaling local fit shape/value mismatch')
    delta=points-center
    coefficient=np.linalg.pinv(delta.T@delta)@(delta.T@(targets-float(anchor_y)))
    errors=np.abs(targets-(float(anchor_y)+delta@coefficient))
    if not np.isfinite(errors).all():raise ValueError('Nonfinite scaling local fit')
    return delta,errors
