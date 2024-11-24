from typing import Generator, List


def stream_names(names) -> Generator[List[str], None, None]:
    yield names[0:3]
    yield names[3:6]
    yield names[6:9]


batch = [("Sajesh", 28),
         ("Rajesh", 26),
         ("Biplab", 23),
         ("Sajesh", 29),
         ("Rajesh", 27),
         ("Biplab", 29),
         ("Sajesh", 30),
         ("Rajesh", 28),
         ("Biplab", 29)]
names, ages = zip(*batch)

batch_names_generator = stream_names(names)
for name, age in zip(batch_names_generator, ages):
    print(f'{name} is {age} years old.')
