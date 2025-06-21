import numpy as np

from . import bisect_ext


def identity(x):
    return x


def pick_nearest(
    frame_list_sorted,
    pick_frame,
    is_lower=False,
    is_larger=False,
    get_index=False,
    key=None,
):
    if key is None:
        key = identity

    try:
        frame_index = bisect_ext.bisect_left(
            frame_list_sorted, x=key(pick_frame), key=key
        )
        key_for_query = key
    except Exception:
        frame_index = bisect_ext.bisect_left(frame_list_sorted, x=pick_frame, key=key)
        key_for_query = identity

    # print(
    #     "frame_index",
    #     frame_index,
    #     "pick_frame",
    #     pick_frame,
    #     frame_list_sorted[frame_index],
    #     frame_list_sorted[frame_index - 1],
    # )

    if frame_index == 0:
        pick_frame = frame_list_sorted[frame_index]
        pick_index = frame_index

    elif frame_index == len(frame_list_sorted):
        pick_frame = frame_list_sorted[-1]
        pick_index = len(frame_list_sorted) - 1

    else:
        pick_frame_pivot = key_for_query(pick_frame)
        # pick_frame_pivot = pick_frame

        if isinstance(pick_frame_pivot, tuple):
            found_v = key(frame_list_sorted[frame_index])
            diff_to_large = 0
            for found_v_el, pick_el in zip(found_v, pick_frame_pivot):
                diff_to_large += found_v_el - pick_el

            diff_to_small = 0
            found_v = key(frame_list_sorted[frame_index - 1])
            for found_v_el, pick_el in zip(found_v, pick_frame_pivot):
                diff_to_small += pick_el - found_v_el
        else:
            diff_to_large = key(frame_list_sorted[frame_index]) - pick_frame_pivot
            diff_to_small = pick_frame - key(frame_list_sorted[frame_index - 1])

        # print(
        #     "diff_to_large",
        #     diff_to_large,
        #     "diff_to_small",
        #     diff_to_small,
        #     "pick frame=",
        #     pick_frame,
        # )

        if diff_to_large == 0:
            pick_frame = frame_list_sorted[frame_index]
            pick_index = frame_index

        elif diff_to_small == 0:
            pick_frame = frame_list_sorted[frame_index]
            pick_index = frame_index

        else:
            # if diff_to_large < diff_to_small or is_larger:
            if is_larger:
                pick_frame = frame_list_sorted[frame_index]
                pick_index = frame_index

            elif is_lower:
                pick_frame = frame_list_sorted[frame_index - 1]
                pick_index = frame_index - 1

            elif diff_to_large < diff_to_small:
                pick_frame = frame_list_sorted[frame_index]
                pick_index = frame_index

            else:
                pick_frame = frame_list_sorted[frame_index - 1]
                pick_index = frame_index - 1

    if get_index:
        return pick_frame, pick_index

    else:
        return pick_frame


def pick_nearest_list(
    frame_list_sorted, pick_frame_list: list, get_index=False, key=None
):
    nearest_values = []
    for frame in pick_frame_list:
        if isinstance(frame, list | np.ndarray):
            nested_nearest_values = pick_nearest_list(
                frame_list_sorted, frame, get_index=get_index, keY=key
            )
            nearest_values.append(nested_nearest_values)
        else:
            nearest_value = pick_nearest(
                frame_list_sorted,
                frame,
                is_lower=False,
                is_larger=False,
                get_index=get_index,
                key=key,
            )
            if get_index:
                nearest_value = nearest_value[1]
            nearest_values.append(nearest_value)

    return nearest_values


def sample_sequence_span_from_pivot_array(
    pivot_array, start_value, end_value, sample_array_list, get_index=False, add_1=True
):
    array_length = len(pivot_array)
    for sample_array in sample_array_list:
        assert len(sample_array) == array_length, (len(sample_array), array_length)

    _, pick_start_index = pick_nearest(
        frame_list_sorted=pivot_array,
        pick_frame=start_value,
        is_lower=True,
        get_index=True,
    )

    if add_1:
        if isinstance(end_value, tuple):
            end_value_search = end_value[:-1] + (end_value[-1] + 1,)

        else:
            end_value_search = end_value + 1
    else:
        end_value_search = end_value

    v, pick_end_index = pick_nearest(
        frame_list_sorted=pivot_array,
        pick_frame=end_value_search,
        is_larger=True,
        get_index=True,
    )
    pick_end_index = max(pick_start_index + 1, pick_end_index)
    # pick_nearestでis_largerだけど、equalなものをとってきた場合は、
    # その次も含むようにしたいので、+1する
    # if end_value in pivot_array:
    # pick_end_index += 1
    # pick_nearestでis_largerだけど、
    # end_value = pivot_array[-1]の場合は、
    # pivot_array[pick_end_index] が、 end_value の一個前になってしまうので、
    if end_value >= pivot_array[-1]:
        pick_end_index += 1

    sample_pivot_array = pivot_array[pick_start_index:pick_end_index]
    sample_span_array_list = []
    for sample_array in sample_array_list:
        sample_span_array_list.append(sample_array[pick_start_index:pick_end_index])

    if get_index:
        return (
            sample_pivot_array,
            sample_span_array_list,
            pick_start_index,
            pick_end_index,
        )

    return sample_pivot_array, sample_span_array_list


def sample_feature_with_min_timespan(
    time_list: list,
    feature_list: np.ndarray,
    timespan: tuple[int | float, int | float],
    require_min_timespan: int | float = 5,
) -> np.ndarray:
    """
    frame_span で与えられた frame の間で feature list を作成する.
    timespan: frame だったり timestamp だったりする. frame_list と対応するものである必要がある
    """
    time_length = timespan[1] - timespan[0]
    if time_length < require_min_timespan:
        mid_time = (timespan[0] + timespan[1]) / 2
        timespan = [
            mid_time - require_min_timespan / 2,
            mid_time + require_min_timespan / 2,
        ]

    _, [sample_feature_list] = sample_sequence_span_from_pivot_array(
        pivot_array=time_list,
        start_value=timespan[0],
        end_value=timespan[1],
        sample_array_list=[feature_list],
        get_index=False,
        add_1=True,
    )

    return sample_feature_list
