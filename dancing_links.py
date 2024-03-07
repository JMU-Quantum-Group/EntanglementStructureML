def hook_right(left_node, right_node):
    left_node.right.left = right_node
    right_node.right = left_node.right
    left_node.right = right_node
    right_node.left = left_node


def hook_down(up_node, down_node):
    up_node.down.up = down_node
    down_node.down = up_node.down
    up_node.down = down_node
    down_node.up = up_node


def unhook_right(node):
    node.right.left = node.left
    node.left.right = node.right


def unhook_down(node):
    node.down.up = node.up
    node.up.down = node.down


def cover(column):
    # Unhook the column header.
    unhook_right(column)

    # Unhook all rows in this column.
    i = column.down
    while i != column:
        i.same.status = 1
        j = i.right
        while j != i:
            unhook_down(j)
            j = j.right
        i = i.down


def cover_same_node(same_node):
    i = same_node.same_next
    while i != same_node:
        if i.status != 1:
            unhook_down(i)
            j = i.right
            while j != i:
                unhook_down(j)
                j = j.right
        i = i.same_next

def uncover_same_node(same_node):
    i = same_node.same_before
    while i != same_node:
        if i.status != 1:
            hook_down(i.up, i)
            j = i.left
            while j != i:
                hook_down(j.up, j)
                j = j.left
        i = i.same_before


def uncover(column):
    # Re-hook all rows in this column.
    i = column.up
    while i != column:
        i.same.status = 0
        j = i.left
        while j != i:
            hook_down(j.up, j)
            j = j.left
        i = i.up

    # Re-hook the column header.
    hook_right(column.left, column)


class Node:
    def __init__(self):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.same_before = self
        self.same_next = self
        self.same = self
        self.status = 0
        self.column = None


class DancingLinks:
    def __init__(self, columns, max_solution=float('inf')):
        self.head = Node()
        self.columns = []
        for i in range(columns):
            column = Node()
            column.column = column
            self.columns.append(column)
            hook_right(self.head.left, column)

        self.max_solution = max_solution

    def append_same_row(self, row_list):
        same_node = None
        for row in row_list:
            start_node = None
            for j in row:
                node = Node()
                column = self.columns[j]

                node.column = j
                column.up.down = node

                node.up = column.up
                node.down = column

                column.up = node
                if start_node:
                    hook_right(start_node.left, node)
                    node.same = start_node
                else:
                    start_node = node

            if same_node:
                start_node.same_before = same_node.same_before
                start_node.same_next = same_node
                same_node.same_before.same_next = start_node
                same_node.same_before = start_node
            else:
                same_node = start_node

    def append_row(self, row):
        start_node = None
        for j in row:
            node = Node()
            column = self.columns[j]

            node.column = j
            column.up.down = node

            node.up = column.up
            node.down = column

            column.up = node
            if start_node:
                hook_right(start_node.left, node)
            else:
                start_node = node

    def search(self):
        if not self.head.right or self.head.right == self.head:
            yield []
            return

        # Choose a column deterministically.
        column = self.head.right.column

        # Cover the chosen column.
        cover(column)

        # Try each row in the chosen column.
        row_node = column.down
        solution_count = 0
        while row_node != column:
            # Add the row to the partial solution.
            solution_row_nodes = [row_node]

            cover_same_node(row_node.same)

            # Cover all other columns with a 1 in this row.
            right_node = row_node.right
            while right_node != row_node:
                cover(self.columns[right_node.column])
                solution_row_nodes.append(right_node)
                right_node = right_node.right

            # Recurse to find more solutions.
            for solution in self.search():
                if solution_count == self.max_solution:
                    break
                solution_count += 1
                yield solution + [[node.column for node in solution_row_nodes]]

            # Uncover all other columns with a 1 in this row.
            left_node = row_node.left
            while left_node != row_node:
                uncover(self.columns[left_node.column])
                left_node = left_node.left

            uncover_same_node(row_node.same)

            # Move to the next row in the chosen column.
            row_node = row_node.down

        # Uncover the chosen column.
        uncover(column)


if __name__ == "__main__":
    # 创建一个 DancingLinks 对象，表示一个 7 列的矩阵。
    dlx = DancingLinks(7)

    # 向矩阵中添加四行。
    dlx.append_row([0, 3, 6])
    dlx.append_same_row([[1, 2], [0, 4], [1, 3]])
    dlx.append_same_row([[2, 4], [1, 5]])
    dlx.append_row([0, 1, 3])
    dlx.append_row([1, 2, 3])
    dlx.append_row([5, 6])
    dlx.append_row([4, 5])

    # 搜索矩阵的所有精确覆盖解。
    solution_list = list()
    for solution in dlx.search():
        solution_list.append(solution)
    print(solution_list)
