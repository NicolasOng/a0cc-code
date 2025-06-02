from cc.ranking import generate_rect_board_lists

a, b = generate_rect_board_lists(4, 4)
for i in range(4):
    print(a[i * 4:(i + 1) * 4])
print("---")
for i in range(4):
    print(b[i * 4:(i + 1) * 4])


x, y = 0, 1
row_major = x * 4 + y
col_major = y * 4 + x

print(row_major, a[row_major], b[a[row_major]])
print(col_major, a[col_major], b[a[col_major]])
print(row_major, b[row_major], a[b[row_major]])
print(col_major, b[col_major], a[b[col_major]])

from a0.wrappers.pywrapper import board_mapping

mapping = board_mapping(4)
print(mapping)