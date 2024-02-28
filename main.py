import streamlit as st
from lida import Manager, TextGenerationConfig, llm
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import networkx as nx
import json

openai.api_key = os.getenv('OPENAI_API_KEY')

lida = Manager(text_gen=llm("openai"))


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)

    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

# Function to process the uploaded files
def process_files(files):
    textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

    data_frames = []
    for file in files:
        df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
        # Append the processed data frame to the list
        data_frames.append(df)

    # Merge the data frames based on the "en" column
    merged_df = pd.concat(data_frames).groupby('en').first().reset_index()

    # Display the merged data frame
    st.write("Merged Data:")
    st.dataframe(merged_df)

    # Display the column labels
    st.write("Column Labels:")
    st.write(merged_df.columns)

    # Perform graph + vector RAG chat query
    en_query = st.text_input("Enter the 'en' value:")
    if en_query:
        results = merged_df[merged_df['en'] == en_query]

        st.write("Query Results:")
        st.dataframe(results)


        # Build the knowledge graph
        G = nx.Graph()
        for _, row in results.iterrows():
            for col in results.columns:
                if col != 'en':
                    value = row[col]
                    if pd.notnull(value):  # Exclude nodes with no values
                        G.add_node(col)
                        G.add_edge(row['en'], col, value=value)

        # Prepare the KG for Cytoscape.js
        cytoscape_elements = []
        for node in G.nodes:
            cytoscape_elements.append({'data': {'id': node, 'label': node}})
        for edge in G.edges:
            source, target = edge
            value = G.edges[edge]['value']
            cytoscape_elements.append({'data': {'source': source, 'target': target, 'value': value}})

        # Convert the KG to JSON
        cytoscape_json = json.dumps(cytoscape_elements)

        # Show the button to open the knowledge graph popup
        if st.button("Open Popup"):
            # Display the KG using Cytoscape.js in a popup
            #st.write("Knowledge Graph:")
            #st.write("Close the popup to continue.")
            html_code = (
                """
                <style>
                    #cy-popup {
                        position: fixed;
                        top: 0;
                        left: 0;
                        height: 100%;
                        width: 100%;
                        z-index: 9999;
                        background-color: rgba(0, 0, 0, 0.4);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }

                    #cy-container {
                        height: 800px;
                        width: 2500px;
                        background-color: white;
                        border-radius: 10px;
                        overflow: hidden;
                    }
                </style>

                <div id="cy-popup">
                    <div id="cy-container"></div>
                </div>

                <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.18.0/cytoscape.min.js"></script>
                <script>
                    var cytoscapeElements = """ + cytoscape_json + """;
                    var cyPopup = document.getElementById('cy-popup');
                    var cyContainer = document.getElementById('cy-container');

                    var cy = cytoscape({
                        container: cyContainer,
                        elements: cytoscapeElements,
                        layout: { name: 'random' },
                        style: [
                            {
                                selector: 'node',
                                style: {
                                    'background-color': 'lightblue',
                                    'label': 'data(label)',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'font-size': '12px'
                                }
                            },
                            {
                                selector: 'edge',
                                style: {
                                    'width': 1,
                                    'line-color': 'black',
                                    'target-arrow-color': 'black',
                                    'target-arrow-shape': 'triangle'
                                }
                            }
                        ]
                    });

                    cyPopup.addEventListener('click', function(event) {
                        if (event.target === cyPopup) {
                            cyPopup.style.display = 'none';
                        }
                    });
                </script>
                """
            )
            st.components.v1.html(html_code, height=700)
            if en_query:
                results = merged_df[merged_df['en'] == en_query]
                csv_file_path = results
                summary = lida.summarize(csv_file_path, summary_method="default", textgen_config=textgen_config)
                # st.write(summary)

                goals = lida.goals(summary, n=2, textgen_config=textgen_config)
                for goal in goals:
                    st.write(goal)

                i = 0
                library = "seaborn"
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)

                charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)
                img_base64_string = charts[0].raster
                img = base64_to_image(img_base64_string)
                st.image(img)

                if len(goals) > 1:
                    second_goal_charts = lida.visualize(summary=summary, goal=goals[1], textgen_config=textgen_config, library=library)
                    second_goal_img_base64_string = second_goal_charts[0].raster
                    second_goal_img = base64_to_image(second_goal_img_base64_string)
                    st.image(second_goal_img)

# Streamlit app
def main():
    st.set_page_config(layout="wide")  # Set the app layout to wide
    st.title("DHV AI Startup Demo Graph Database Querying")

    st.write("Upload your Excel/CSV files")

    # Allow user to uploadmultiple files
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['csv', 'xlsx'])

    if uploaded_files:
        # Process the uploaded files
        process_files(uploaded_files)

if __name__ == "__main__":
    main()