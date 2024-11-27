def test1():
    import re
    # Example Markdown Document with Multiple Tables
    markdown_document = """
    # Section 1
    Some text here.

    | Header A       | Header B       |
    |----------------|----------------|
    | Row 1A         | Row 1B         |
    | Row 2A         | Row 2B         |

    # Section 2
    More text.

    | Header 1       | Header 2       | Header 3       |
    |-----------------|----------------|----------------|
    | Row 1, Column 1| Row 1, Column 2| Row 1, Column 3|
    | Row 2, Column 1| Row 2, Column 2| Row 2, Column 3|
    | Row 3, Column 1| Row 3, Column 2| Row 3, Column 3|

    Some final text here.
    """

    def extract_tables(markdown):
        # Updated regex to capture Markdown tables
        table_regex = r"(?:\|.+\|\n)+\|[-:\s]+?\|\n(?:\|.+\|\n)+"
        tables = re.findall(table_regex, markdown)
        return tables

    def parse_markdown_table(table):
        # Regex to capture rows
        rows = re.findall(r"^\|.*?\|$", table, re.MULTILINE)

        # Process each row to extract cells
        parsed_table = []
        for row in rows:
            cells = [cell.strip() for cell in row.strip('|').split('|')]
            parsed_table.append(cells)

        # Filter out divider rows
        parsed_table = [row for row in parsed_table if not all(
            cell.strip('- ') == '' for cell in row)]
        return parsed_table

    # Detect tables in the document
    tables = extract_tables(markdown_document)

    # Parse each table
    for i, table in enumerate(tables):
        print(f"Table {i+1}:")
        parsed_table = parse_markdown_table(table)
        for row in parsed_table:
            print(row)
        print()


def test2():
    from pathlib import Path
    import streamlit as st

    page_name = st.text_input("Page name", "page2")

    if page_name and st.button("Create new page"):
        (Path("pages") / f"{page_name}.py").write_text(
            f"""
    import streamlit as st

    st.write("Hello from {page_name}")
    """,
            encoding="utf-8",
        )


def test3():
    text = "abcadef"
    op = text.split('a')
    print(open)


def test4(table):
    import markdown
    import pandas as pd
    import streamlit as st
    from st_aggrid import AgGrid, GridOptionsBuilder
    import re
    from utils.data_formatting import extract_markdown_tables

    html_content, dataframes = extract_markdown_tables(table)
    pattern = r'(<div id="aggrid-table-(\d+)"></div>)'

    # Initialize variables
    last_pos = 0  # Track the last position of the processed string

    # Step 3: Iterate over matches and replace divs with AgGrid
    for match in re.finditer(pattern, html_content):
        # Add content before the current placeholder
        st.markdown(html_content[last_pos:match.start()],
                    unsafe_allow_html=True)

        # Get the table index from the div ID
        table_index = int(match.group(2))

        if 0 <= table_index < len(dataframes):
            # Render AgGrid for the corresponding DataFrame
            grid_builder = GridOptionsBuilder.from_dataframe(
                dataframes[table_index])
            grid_builder.configure_selection(
                selection_mode="multiple", use_checkbox=True)
            grid_builder.configure_side_bar(
                filters_panel=True, columns_panel=False)
            grid_options = grid_builder.build()
            AgGrid(data=dataframes[table_index],
                   gridOptions=grid_options, key=f'grid{table_index}')

        # Update the last processed position
        last_pos = match.end()
    st.markdown(html_content[last_pos:], unsafe_allow_html=True)

    # Step 4: Render remaining static HTML (after all AgGrids)
    # st.markdown(''.join(modified_html), unsafe_allow_html=True)

    # # Call the function to replace placeholders
    # replace_with_aggrid(html_content, dataframes)


def test5():
    from st_aggrid import AgGrid
    import pandas as pd

    df = pd.read_json(
        "https://www.ag-grid.com/example-assets/olympic-winners.json")
    grid_return = AgGrid(df)


table = """
This is an example of text embedded with table.

Lets see if the regex can find table in this case?

Lets see!

| ID  | Name       | Age | Occupation         | City           | Country      | Email                 | Phone       | Company           | Salary    |
|-----|------------|-----|--------------------|----------------|--------------|-----------------------|-------------|-------------------|-----------|
| 1   | Alice      | 30  | Software Engineer  | New York       | USA          | alice@example.com     | 123-456-7890| TechCorp          | $120,000  |
| 2   | Bob        | 25  | Graphic Designer   | Los Angeles    | USA          | bob@example.com       | 987-654-3210| Creative Studio   | $80,000   |
| 3   | Charlie    | 35  | Teacher            | Chicago        | USA          | charlie@example.com   | 555-555-5555| Local School      | $70,000   |
| 4   | Diana      | 28  | Data Scientist     | San Francisco  | USA          | diana@example.com     | 444-444-4444| DataAnalytics Inc | $115,000  |
| 5   | Ethan      | 40  | Product Manager    | Seattle        | USA          | ethan@example.com     | 333-333-3333| InnovateTech      | $130,000  |
| 6   | Fiona      | 27  | UX Designer        | Austin         | USA          | fiona@example.com     | 222-222-2222| DesignLab         | $85,000   |
| 7   | George     | 50  | CEO                | Boston         | USA          | george@example.com    | 111-111-1111| GlobalCorp        | $250,000  |
| 8   | Hannah     | 22  | Intern             | Miami          | USA          | hannah@example.com    | 666-666-6666| StartupHub        | $40,000   |
| 9   | Ian        | 45  | Financial Analyst  | Denver         | USA          | ian@example.com       | 777-777-7777| FinanceGroup      | $100,000  |
| 10  | Julia      | 33  | HR Manager         | Atlanta        | USA          | julia@example.com     | 888-888-8888| HR Solutions      | $90,000   |

Let me know if this answers your query!
Here is another table.

| ID  | Name       | Age |
|-----|------------|-----|
| 1   | Alice      | 30  |
| 2   | Bob        | 25  |
| 3   | Charlie    | 35  |
| 4   | Diana      | 28  |
| 5   | Ethan      | 40  |
| 6   | Fiona      | 27  |
"""
# test4(table=table)


def test6():
    from st_aggrid import AgGrid, GridOptionsBuilder
    import pandas as pd
    df = pd.read_json(
        "https://www.ag-grid.com/example-assets/olympic-winners.json")
    grid_builder = GridOptionsBuilder.from_dataframe(df)
    sidebar_options = {
        "sideBar": {
            "toolPanels": [
                "filters",
                {
                    "id": 'columns',
                    "labelDefault": 'Columns',
                    "labelKey": 'columns',
                    "iconKey": 'columns',
                    "toolPanel": 'agColumnsToolPanel',
                    "toolPanelParams": {
                        "suppressPivotMode": True,
                        "suppressRowGroups": True,
                        "suppressValues": True

                    }
                }
            ]
        },
    }

    grid_builder.configure_grid_options(**sidebar_options)
    # grid_builder.configure_side_bar()
    grid_options = grid_builder.build()

    AgGrid(data=df, gridOptions=grid_options,
           key='grid2')


def test7():
    from utils.data_formatting import extract_material_ids

    text = 'What are the physical properties of materials Mp-121 and Mp-31?'
    y = extract_material_ids(text)
    print(y)


test7()
