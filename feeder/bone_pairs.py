ntu_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)

hdm_pairs = (
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (1, 12), (12, 13),
        (13, 14), (14, 14), (14, 15), (15, 16), (16, 17), (18, 14),
        (14, 25), (18, 19), (19, 20), (20, 21), (21, 22), (21, 24),
        (22, 23), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (31, 28))

ntu_for_fourier = tuple((i-1, j-1) for (i,j) in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (23, 8), (24, 25),(25, 12)
))