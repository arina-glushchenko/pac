class Item:
    def __init__(self, count=3, max_count=16):
        self._count = count
        self._max_count = 16

    def update_count(self, val):
        if val <= self._max_count:
            self._count = val
            return True
        else:
            return False

    def __add__(self, num):
        """ Сложение с числом """
        if (num + self.count) <= self._max_count:
            return self.count + num
        else:
            print("Значение превышает maxcount")
            return self

    def __sub__(self, num):
        """ Вычитание числа """
        if (self.count - num) >= 0:
            return self.count - num
        else:
            print("Значение превышает maxcount")
            return self

    def __mul__(self, num):
        """ Умножение на число """
        if (num * self.count) <= self._max_count:
            return self.count * num
        else:
            print("Значение превышает maxcount")
            return self

    def __lt__(self, num):
        """ Сравнение меньше """
        return self.count < num

    def __gt__(self, num):
        """ Сравнение больше """
        return self.count > num

    def __le__(self, num):
        """Сравнение меньше или равно"""
        return self.count <= num

    def __ge__(self, num):
        """Сравнение больше или равно"""
        return self.count >= num

    def __eq__(self, num):
        """Сравнение равно равно"""
        return self.count == num

    def __iadd__(self, num):
        """ Присваивание и сложение """
        if (self._count + num) <= self._max_count:
            self._count += num
            return self
        else:
            print("Значение превышает maxcount")
            return self

    def __isub__(self, num):
        """ Присваивание и вычитание """
        if (self._count - num) >= 0:
            self._count -= num
            return self
        else:
            print("Значение меньше нуля")
            return self

    def __imul__(self, num):
        """ Присваивание и умножение """
        if (self._count * num) <= self._max_count:
            self._count *= num
            return self
        else:
            print("Число превышает maxcount, операция не выполнена")
            return self
    @property
    def count(self):
        return self._count

class Fruit(Item):
    def __init__(self, ripe=True, **kwargs):
        super().__init__(**kwargs)
        self._ripe = ripe

class Food(Item):
    def __init__(self, saturation, **kwargs):
        super().__init__(**kwargs)
        self._saturation = saturation

    @property
    def eatable(self):
        return self._saturation > 0


class Apple(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color


class Kiwi(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='green', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

class Orange(Fruit, Food):
    def __init__(self, ripe, count=1, max_count=32, color='orange', saturation=10):
        super().__init__(saturation=saturation, ripe=ripe, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

class Tomato(Food):
    def __init__(self, count=1, max_count=32, color='red', saturation=10):
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color

class Potato(Food):
    def __init__(self, count=1, max_count=32, color='purple', saturation=10):
        super().__init__(saturation=saturation, count=count, max_count=max_count)
        self._color = color

    @property
    def color(self):
        return self._color



class Inventory(Food):
    def __init__ (self):
        self.list = [None] * 10

    def __getitem__(self, index):
        """ Получение элемента по индексу """
        if index > len(self.list):
            raise IndexError(f'Index {index} more then {len(self)}')
        return self.list[index]

    def add_object(self, index, Item):
        if (index > len(self.list) or index < 0):
            print("Индекс не соответствует параметрам")
        elif self.list[index] == None:
            if Item.eatable:
                self.list[index] = Item
        else:
            print("Эта ячейка уже занята")

    def remove_object(self, index, count):
        if count > (self.list[index]._count):
            print("Вы хотите удалить слишком много объектов")
        elif count < (self.list[index]._count):
            self.list[index]._count -= count
        else:
            self.list[index] = None

