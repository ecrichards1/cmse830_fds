# March Madness Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff

# Setting up page
st.set_page_config(page_title="March Madness Data App", layout="wide")
st.title("March Madness Data App")

# Loading in data
path = "DEV _ March Madness.csv"
mm = pd.read_csv(path)

# Summary Statistics; Data Overview
st.subheader("Dataset Overview")
st.write("Shape:", mm.shape)
st.dataframe(mm.head())
st.write("Summary Statistics:")
st.write(mm.describe())

# Data cleaning
mm_sub = mm.loc[mm['Tournament Winner?'] == 'No', # Conference winners automatically qualify, so I excluded them, as the goal is predicting tournament fates of at-large teams
                ['Season','Mapped ESPN Team Name','Adjusted Offensive Efficiency','Adjusted Defensive Efficiency','eFGPct','TOPct',
                 'AdjTempo','PGOR','PGDR','Tournament Winner?','Post-Season Tournament']]

mm_sub = mm_sub[['Season','Mapped ESPN Team Name','Adjusted Offensive Efficiency','Adjusted Defensive Efficiency','eFGPct','TOPct',
                 'AdjTempo','PGOR','PGDR','Post-Season Tournament']] # There was over 150 variables, so I decided to cut this analysis down a bit, but I plan to do more on the full project.

mm_sub = mm_sub[(mm_sub['eFGPct'] != 0) & (mm_sub['Adjusted Defensive Efficiency'] != 0)] # Some teams didn't play 2020-21 due to COVID, so this filters them out.

mm_sub.rename(columns={
    'Adjusted Offensive Efficiency': 'AdjOE',
    'Adjusted Defensive Efficiency': 'AdjDE',
    'Post-Season Tournament': 'March Madness?',
    'Mapped ESPN Team Name': 'School' # This changes the variable names so they're more simple to access; I'd rather not type out the full 3 words when 5 letters would do fine.
}, inplace=True)

mm_sub['March Madness?'] = (mm_sub['March Madness?'] == 'March Madness').astype(int) # Turns March Madness into 0 and 1; scanning to see if the tournament the team made was March Madness.

# Missing Values Visualization
st.subheader("Missing Values Plot")

nan_mask = mm_sub.isna()
nan_array = nan_mask.astype(int).to_numpy()

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')
ax.set_xlabel('MM Index')
ax.set_ylabel('Features')
ax.set_title('Visualizing Missing Values in MM Subset')
ax.set_yticks(range(len(mm_sub.columns)))
ax.set_yticklabels(mm_sub.columns)
num_mm = nan_array.shape[0]
ax.set_xticks(np.linspace(0, num_mm-1, min(10, num_mm)).astype(int))
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

# All seasons before 2007 had no values for PGOR or PGDR
mm_sub = mm_sub[mm_sub['Season'] > 2006]

st.markdown("""
### Variable Explanation
- **AdjOE**: Adjusted Offensive Efficiency — points scored per 100 possessions, adjusted for opponent
- **AdjDE**: Adjusted Defensive Efficiency — points allowed per 100 possessions, adjusted for opponent
- **eFGPct**: Effective Field Goal Percentage (Formula listed below)
- **TOPct**: Turnover Percentage — Percentage of possessions ending in a turnover
- **AdjTempo**: Adjusted Pace — Possessions per 40 minutes
- **PGOR / PGDR**: Offensive and Defensive Ratings for Point Guards, respectively.
- **March Madness?**: 1 if the team qualified for March Madness, 0 if not
""")

# Latex formula for EFG%
st.latex(r"\text{eFG\%} = \frac{FGM + 0.5 \times 3PM}{FGA}")

# Correlation Heatmap
st.subheader("Correlation Heatmap")

selected_features = ['eFGPct','TOPct','AdjTempo','AdjOE','AdjDE','PGOR','PGDR','March Madness?']
selected_features = [f for f in selected_features if f in mm_sub.columns]

correlation_matrix = mm_sub[selected_features].corr().round(3).values

fig_heatmap = ff.create_annotated_heatmap(
    z=correlation_matrix,
    x=selected_features,
    y=selected_features,
    colorscale='bluered',
    showscale=True
)
fig_heatmap.update_layout(
    title="Correlation Heatmap (Interactive)",
    xaxis_title="Features",
    yaxis_title="Features",
    width=700,
    height=700
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Interactive histogram
st.subheader("Histograms for Variables")

opts = ['AdjOE','AdjDE','eFGPct','TOPct',
    'AdjTempo','PGOR','PGDR','March Madness?'] # Defines options for dropdown menus

plot_col = st.selectbox(
    "Select a variable to plot:",
    options=opts
)

fig_hist = px.histogram(mm_sub, x=plot_col, nbins=30, title=f"Histogram of {plot_col}")
st.plotly_chart(fig_hist, use_container_width=True)

# Scatterplot; x and y can be chosen
st.subheader("Customizable Scatter Plot")

# Dropdowns for x and y axes
x_sc = st.selectbox("Select X-axis variable:", options=opts, index=0)
y_sc = st.selectbox("Select Y-axis variable:", options=opts, index=1)

# Creates scatter plot with color coding (green if made tournament, red if not)
fig_scatter = px.scatter(
    mm_sub,
    x=x_sc,
    y=y_sc,
    color=mm_sub['March Madness?'].map({1: 'Made NCAA Tournament', 0: 'Did Not Qualify'}),
    color_discrete_map={
        'Made NCAA Tournament': 'green', # Color codes teams according to qualification or lack thereof
        'Did Not Qualify': 'red'
    },
    title=f"{x_sc} vs {y_sc}",
    hover_data=['Season', 'School'] # Shows which team it was and the year, e.g. 2023 Kentucky, 2019 Michigan, etc.
)

st.plotly_chart(fig_scatter, use_container_width=True)

#streamlit run "C:/Users/ecric/OneDrive/Desktop/mm_streamlit.py"
# Above is to keep handy the command to test the app.