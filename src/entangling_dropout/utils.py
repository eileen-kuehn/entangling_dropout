def batch_generator(x_values, y_values, batch_size, rd):
    permutation = rd.permutation(range(len(x_values)))
    current_position = 0
    while current_position < len(x_values):
        try:
            yield x_values[
                permutation[current_position : current_position + batch_size]
            ], y_values[permutation[current_position : current_position + batch_size]]
        except IndexError:
            yield x_values[permutation[current_position:]], y_values[
                permutation[current_position:]
            ]
        current_position += batch_size
