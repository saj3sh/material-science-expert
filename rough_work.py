import json
from typing import get_origin, get_args, Union
from typing import get_origin, get_args, Optional
from pydantic import BaseModel
from typing import get_origin, get_args
from pydantic import BaseModel, Field
from typing import Any, Dict, Type
from typing import Generator, List
from emmet.core.summary import SummaryDoc


def stream_names(names) -> Generator[List[str], None, None]:
    yield names[0:3]
    yield names[3:6]
    yield names[6:9]


def test1():
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


descriptions = []


def fetch_field_descriptions(model: type[BaseModel]) -> dict:
    for field_name, field_info in model.model_fields.items():
        field_description = field_info.description
        field_annotation = field_info.annotation

        # Determine the origin and arguments of the field's annotation
        origin = get_origin(field_annotation)
        args = get_args(field_annotation)

        # Unwrap Union (for Optional/Union[X, None] types)
        if origin is Union and type(None) in args:
            # Remove None from Union
            field_annotation = next(
                arg for arg in args if arg is not type(None))
            origin = get_origin(field_annotation)
            args = get_args(field_annotation)

        # Check if the field is a list of Pydantic models
        if origin is list and args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            descriptions.append(
                {"field": field_name, "description": field_description})
            fetch_field_descriptions(args[0])
        # Check if the field is a Pydantic model
        elif isinstance(field_annotation, type) and issubclass(field_annotation, BaseModel):
            descriptions.append(
                {"field": field_name, "description": field_description})
            fetch_field_descriptions(field_annotation)
        else:
            descriptions.append(
                {"field": field_name, "description": field_description})
    return descriptions


def test2():
    x = SummaryDoc.model_json_schema()
    y = fetch_field_descriptions(SummaryDoc)
    jsonStr = json.dumps(y)
    print(x)


test2()
