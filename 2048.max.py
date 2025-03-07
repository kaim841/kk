import random
import numpy as np
import tkinter as tk
from tkinter import messagebox


class Game2048:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('2048 Game')
        self.grid_cells = []
        self.matrix = np.zeros((4, 4), dtype=int)
        self.score = 0

        # 创建GUI网格
        background = tk.Frame(self.window, bg='#92877d')
        background.grid()

        for i in range(4):
            grid_row = []
            for j in range(4):
                cell = tk.Frame(
                    background,
                    bg='#9e948a',
                    width=100,
                    height=100
                )
                cell.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(
                    cell,
                    bg='#9e948a',
                    justify=tk.CENTER,
                    font=('Arial', 30, 'bold'),
                    width=4,
                    height=2
                )
                cell_number.grid()
                grid_row.append(cell_number)
            self.grid_cells.append(grid_row)

        # 绑定按键事件
        self.window.bind('<Left>', lambda e: self.move('left'))
        self.window.bind('<Right>', lambda e: self.move('right'))
        self.window.bind('<Up>', lambda e: self.move('up'))
        self.window.bind('<Down>', lambda e: self.move('down'))

        # 初始化游戏
        self.init_game()

    def init_game(self):
        """初始化游戏"""
        self.matrix = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()
        self.update_grid()

    def add_new_tile(self):
        """在空位置随机添加新数字(2或4)"""
        empty_cells = [(i, j) for i in range(4) for j in range(4)
                       if self.matrix[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.matrix[i][j] = random.choice([2, 4])

    def update_grid(self):
        """更新GUI显示"""
        for i in range(4):
            for j in range(4):
                value = self.matrix[i][j]
                if value == 0:
                    self.grid_cells[i][j].configure(
                        text='',
                        bg='#9e948a'
                    )
                else:
                    self.grid_cells[i][j].configure(
                        text=str(value),
                        bg=self.get_cell_color(value),
                        fg=self.get_number_color(value)
                    )
        self.window.update_idletasks()

    def get_cell_color(self, value):
        """获取方块背景色"""
        colors = {
            0: '#9e948a',
            2: '#eee4da',
            4: '#ede0c8',
            8: '#f2b179',
            16: '#f59563',
            32: '#f67c5f',
            64: '#f65e3b',
            128: '#edcf72',
            256: '#edcc61',
            512: '#edc850',
            1024: '#edc53f',
            2048: '#edc22e'
        }
        return colors.get(value, '#ff0000')

    def get_number_color(self, value):
        """获取数字颜色"""
        return '#776e65' if value <= 4 else '#f9f6f2'

    def compress(self, matrix):
        """压缩数组，移除零"""
        new_matrix = np.zeros((4,), dtype=int)
        pos = 0
        for i in range(4):
            if matrix[i] != 0:
                new_matrix[pos] = matrix[i]
                pos += 1
        return new_matrix

    def merge(self, matrix):
        """合并相同数字"""
        score = 0
        for i in range(3):
            if matrix[i] != 0 and matrix[i] == matrix[i + 1]:
                matrix[i] *= 2
                score += matrix[i]
                matrix[i + 1] = 0
        return matrix, score

    def move(self, direction):
        """处理移动"""
        moved = False
        score = 0

        if direction == 'left':
            for i in range(4):
                row = self.matrix[i]
                new_row = self.compress(row)
                new_row, new_score = self.merge(new_row)
                new_row = self.compress(new_row)
                score += new_score
                if not np.array_equal(row, new_row):
                    moved = True
                    self.matrix[i] = new_row

        elif direction == 'right':
            for i in range(4):
                row = self.matrix[i][::-1]
                new_row = self.compress(row)
                new_row, new_score = self.merge(new_row)
                new_row = self.compress(new_row)
                score += new_score
                if not np.array_equal(row, new_row):
                    moved = True
                    self.matrix[i] = new_row[::-1]

        elif direction == 'up':
            for j in range(4):
                col = self.matrix[:, j]
                new_col = self.compress(col)
                new_col, new_score = self.merge(new_col)
                new_col = self.compress(new_col)
                score += new_score
                if not np.array_equal(col, new_col):
                    moved = True
                    self.matrix[:, j] = new_col

        elif direction == 'down':
            for j in range(4):
                col = self.matrix[::-1, j]
                new_col = self.compress(col)
                new_col, new_score = self.merge(new_col)
                new_col = self.compress(new_col)
                score += new_score
                if not np.array_equal(col, new_col):
                    moved = True
                    self.matrix[::-1, j] = new_col

        if moved:
            self.score += score
            self.add_new_tile()
            self.update_grid()
            if self.game_over():
                messagebox.showinfo('Game Over', f'Final Score: {self.score}')
                self.init_game()

    def game_over(self):
        """检查游戏是否结束"""
        if 0 in self.matrix:
            return False

        for i in range(4):
            for j in range(4):
                value = self.matrix[i][j]
                if (i < 3 and value == self.matrix[i + 1][j]) or \
                        (j < 3 and value == self.matrix[i][j + 1]):
                    return False
        return True

    def run(self):
        """运行游戏"""
        self.window.mainloop()


if __name__ == '__main__':
    game = Game2048()
    game.run()