from collections import deque


def count_islands(grid, rows, cols):
    visited = [[False] * cols for _ in range(rows)]
    islands = 0

    directions = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]

    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()

            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not visited[nr][nc]
                    and grid[nr][nc] == 1
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                islands += 1
                bfs(r, c)

    return islands


def read_dimensions():
    print("Enter matrix dimensions: rows columns")
    print("Example: 3 4")

    raw = input("> ").strip().split()

    if len(raw) != 2:
        raise ValueError("Dimensions must contain exactly 2 numbers: rows and columns.")

    try:
        rows, cols = map(int, raw)
    except ValueError:
        raise ValueError("Dimensions must be integers.")

    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive integers.")

    return rows, cols


def read_matrix(rows, cols):
    print(f"Enter matrix: {rows} rows, each with {cols} values.")
    print("Allowed values: 0 = ocean, 1 = island")

    grid = []

    for r in range(rows):
        raw_row = input(f"Row {r + 1}> ").strip().split()

        if len(raw_row) != cols:
            raise ValueError(
                f"Row {r + 1} must contain exactly {cols} values, "
                f"but got {len(raw_row)}."
            )

        try:
            row = list(map(int, raw_row))
        except ValueError:
            raise ValueError(f"Row {r + 1} contains non-integer values.")

        invalid_values = [value for value in row if value not in (0, 1)]

        if invalid_values:
            raise ValueError(
                f"Row {r + 1} contains invalid values: {invalid_values}. "
                "Only 0 and 1 are allowed."
            )

        grid.append(row)

    return grid


def main():
    try:
        rows, cols = read_dimensions()
        grid = read_matrix(rows, cols)

        result = count_islands(grid, rows, cols)
        print(f"\n Number of islands: {result}")

    except ValueError as error:
        print(f"Input error: {error}")


if __name__ == "__main__":
    main()