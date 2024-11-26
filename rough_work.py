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


def test3():
    import re
    pattern = r"mp-\d+"

    text = "Here are examples: mp-12345,MP-6789,mP-101, and Mp-999."
    matches = re.findall(pattern, text, re.IGNORECASE)
    print(matches)


def test4():
    from st_aggrid import AgGrid
    import pandas as pd

    df = pd.read_csv(
        'https://raw.githubusercontent.com/fivethirtyeight/data/master/airline-safety/airline-safety.csv')
    AgGrid(df)


def test5():
    import streamlit as st
    import numpy as np
    import pandas as pd

    from st_aggrid import AgGrid, GridOptionsBuilder

    @st.cache_data()
    def get_data():
        df = pd.DataFrame(
            np.random.randint(0, 100, 50).reshape(-1, 5), columns=list("abcde")
        ).applymap(str)
        return df

    data = get_data()

    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_side_bar(filters_panel=True, columns_panel=False)
    go = gb.build()

    AgGrid(
        data,
        gridOptions=go,
        key='an_unique_key'
    )


def test6():
    import streamlit as st

    st.markdown("""
    <style>
        .flex-container {
        display: flex;  
        }

        .flex-child {
            flex: 1
        }  

        .flex-child:first-child {
            margin-right: 20px;
        } 

        p {
            position: absolute;
            bottom:0;
            text-align: center;
        }


    </style>

    <div class="flex-container">
        <div class="flex-child">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><!--! Font Awesome Pro 6.1.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license (Commercial License) Copyright 2022 Fonticons, Inc. --><path d="M288 256C288 273.7 273.7 288 256 288C238.3 288 224 273.7 224 256C224 238.3 238.3 224 256 224C273.7 224 288 238.3 288 256zM0 256C0 114.6 114.6 0 256 0C397.4 0 512 114.6 512 256C512 397.4 397.4 512 256 512C114.6 512 0 397.4 0 256zM325.1 306.7L380.6 162.4C388.1 142.1 369 123.9 349.6 131.4L205.3 186.9C196.8 190.1 190.1 196.8 186.9 205.3L131.4 349.6C123.9 369 142.1 388.1 162.4 380.6L306.7 325.1C315.2 321.9 321.9 315.2 325.1 306.7V306.7z"/></svg>
        </div>
        <div class="flex-child">
            <p> 
            Examples moved to: <br> <a href="https://pablocfonseca-streamlit-aggrid-examples-example-jyosi3.streamlitapp.com/">https://pablocfonseca-streamlit-aggrid-examples-example-jyosi3.streamlitapp.com/</a>
            </p>
        </div>
    </div>

    """, unsafe_allow_html=True)


test5()
