#pylint: disable-all

class Greeting():
    def __init__(self, message="joe doe"):
        self.message_ = message

    def say_hello(self):
        print(f"hello you {self.message_}")

def greeting(my_greeting = "hey", name = "joe"):
    print(f"{my_greeting} {name}")


if __name__ == "__main__":
    pass

