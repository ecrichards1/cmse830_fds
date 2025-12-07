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
tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Exploratory Data Analysis", "Prediction Tool", "Historical Search"])

# Loading in data
path = "DEV _ March Madness.csv"
#path = "C:/Users/ecric/OneDrive/Desktop/March.csv" # This line is for testing the code in VS Code.
mm = pd.read_csv(path)

with tab1:
    st.subheader("Problem Definition")
    st.markdown("""
    **March Madness**, the NCAA Mens Basketball Tournament, is one of the most  
    unpredictable and widely analyzed sporting events in the United States.  
    Every March, 68 teams compete in a single-elimination bracket to determine  
    the national champion.

    While 32 teams receive **automatic bids** by winning their conference  
    tournaments, the remaining **at-large bids** are selected by the NCAA  
    Selection Committee. These selections depend on a combination of metrics,  
    advanced analytics, résumé strength, and the "eye test".

    ### Why This Matters
    Fans, analysts, and teams themselves constantly debate:
    - **Which teams deserve to make the tournament?**
    - **What statistical profiles best predict qualification?**
    - **How important are offensive efficiency, defensive efficiency, pace, or shooting metrics?**
    - **Are certain teams under- or over-seeded based on analytics?**

    Even after teams are selected, another major debate begins:
    - **Do certain statistical profiles correspond to better seeds?**
    - **Are there measurable differences between tournament teams and those that miss the cut?**

    ### App Purpose
    This app explores:
    1. **Which statistical features are most associated with making March Madness**
    2. **How various basketball efficiency metrics relate to tournament qualification**
    3. **Whether we can build a simple predictive model for NCAA Tournament bids**
    4. **How a team’s statistical profile compares to similar historical teams**

    """)

with tab2:
    # The data came from here: https://www.kaggle.com/datasets/jonathanpilafas/2024-march-madness-statistical-analysis/data
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

    mm_sub = mm_sub[(mm_sub['eFGPct'] != 0) & (mm_sub['Adjusted Defensive Efficiency'] != 0)]  # Some teams didn't play 2020-21 due to COVID, so this filters them out. The 2020 tournament was also canceled.

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
    mm_sub = mm_sub[(mm_sub['Season'] > 2006) & (mm_sub['Season'] != 2020)]

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

with tab3:

    st.header("Prediction Tool")

    st.write("Select which variables you'd like to use to predict whether a team makes March Madness.")

    feature_options = ['AdjOE','AdjDE','eFGPct','TOPct','AdjTempo','PGOR','PGDR'] # All possible variables

    selected_features = st.multiselect( # User chooses features
        "Choose features for the prediction model:",
        feature_options,
        default=['AdjOE','AdjDE','eFGPct']
    )

    
    if len(selected_features) == 0:
        st.warning("Please select at least one feature to build a model.") # Requires at least one feature; we can't have a blank model with no vars
    else:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        
        X = mm_sub[selected_features] # Preparing data; grabbing the chosen features
        y = mm_sub['March Madness?']

        
        X_train, X_test, y_train, y_test = train_test_split(  # Train/test split data for evaluation
            X, y, test_size=0.2, random_state=42
        )

        
        scaler = StandardScaler() # Scaling
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        
        model = LogisticRegression() # Model training/fitting
        model.fit(X_train_scaled, y_train)

       
        accuracy = model.score(X_test_scaled, y_test) # Testing accuracy with model.score
        st.write(f"Model Accuracy: **{accuracy:.3f}**")

        st.subheader("Enter Team Stats to Predict Tournament Odds")

        
        user_inputs = {} # Create numeric inputs for selected features
        for feature in selected_features:
            default_value = float(mm_sub[feature].mean())
            user_inputs[feature] = st.number_input(
                f"{feature}:",
                value=default_value
            )

        # Converting input into array
        input_values = np.array([list(user_inputs.values())])
        input_scaled = scaler.transform(input_values)

        # Prediction button
        if st.button("Predict"):

            # Model predictions
            pred_prob = model.predict_proba(input_scaled)[0, 1]
            pred_label = model.predict(input_scaled)[0]

            # Output prediction
            if pred_label == 1:
                st.success(f"Prediction: **This team WOULD make the NCAA Tournament.**")
            else:
                st.error(f"Prediction: **This team would NOT make the NCAA Tournament.**")

            st.write(f"Probability of making the tournament: **{pred_prob:.3f}**")

            # ---- MOST SIMILAR TEAMS ----
            st.subheader("Most Similar Teams")

            # Scale entire dataset with same scaler
            X_all_scaled = scaler.transform(X)

            # Get 1D scaled input
            user_scaled = input_scaled[0]

            # Compute Euclidean distance for each team
            distances = np.linalg.norm(X_all_scaled - user_scaled, axis=1)

            # Indices of top 3 closest historical teams
            nearest_idx = np.argsort(distances)[:3]

            similar_teams = mm_sub.iloc[nearest_idx].copy()
            similar_teams["Distance"] = distances[nearest_idx]

            # Display
            for rank, (_, row) in enumerate(similar_teams.iterrows(), start=1):

                team = row["School"]
                season = row["Season"]
                made = "Made Tournament" if row["March Madness?"] == 1 else "Missed Tournament"
                dist = row["Distance"]

                st.markdown(f"""
                ### Similar Team #{rank}
                **{team} ({season})**  
                - Outcome: **{made}**  
                - Similarity Score (lower = more similar): `{dist:.3f}`
                """)

with tab4:
    st.header("Historical Explorer")

    st.markdown("""
    Select a **team** and **season** to explore its statistical profile,
    the model-estimated March Madness probability, and the three most
    similar teams from history.
    """)

    hist_features = [
        'AdjOE', 'AdjDE', 'eFGPct', 'TOPct',
        'AdjTempo', 'PGOR', 'PGDR'
    ]

    available_features = [f for f in hist_features if f in mm_sub.columns]

    # Drop any rows missing required variables
    data4 = mm_sub.dropna(subset=available_features + ["March Madness?"]).copy()

    # Train a separate logistic regression model
    from sklearn.linear_model import LogisticRegression

    X4 = data4[available_features].values
    y4 = data4["March Madness?"].values

    model4 = LogisticRegression(max_iter=500)
    model4.fit(X4, y4)

    # --------------------------
    # TEAM + SEASON SELECTION
    # --------------------------
    team_list = sorted(data4['School'].unique())
    selected_team = st.selectbox("Select Team", team_list)

    season_list = sorted(data4['Season'].unique())
    selected_season = st.selectbox("Select Season", season_list)

    # Extract row for selected team
    row = data4[(data4["School"] == selected_team) &
                (data4["Season"] == selected_season)]

    if row.empty:
        st.warning("No data available for this team and season.")
    else:
        row = row.iloc[0]

        st.subheader(f"{selected_team} — {selected_season}")

        # Metrics Table
        metrics_df = row[available_features].to_frame("Value")
        st.dataframe(metrics_df)

        # Qualification
        made = "Yes" if row["March Madness?"] == 1 else "No"
        st.markdown(f"### Made Tournament: **{made}**")

 
        # Model Prediction
        X_row = row[available_features].values.reshape(1, -1)
        pred_prob = model4.predict_proba(X_row)[0][1]

        st.markdown(f"### Model-Estimated Probability: **{pred_prob:.3f}**")

        # Similarity Search

        def euclid_dist(r2):
            return np.linalg.norm(row[available_features].values -
                                  r2[available_features].values)

        temp = data4.copy()
        temp["dist"] = temp.apply(euclid_dist, axis=1)

        # Exclude the identical team/season
        comps = temp[(temp["School"] != selected_team) |
                     (temp["Season"] != selected_season)]

        comps = comps.sort_values("dist").head(3)

        st.subheader("Most Similar Historical Teams")

        for _, r in comps.iterrows():
            status = "Yes" if r["March Madness?"] == 1 else "No"
            st.markdown(f"- **{r['School']} ({r['Season']})** — Made Tournament: **{status}**")

        # Comparison Bar
        st.subheader("Comparison with Closest Match")

        closest = comps.iloc[0]

        comp_df = pd.DataFrame({
            "Metric": available_features,
            selected_team: row[available_features].values,
            closest["School"]: closest[available_features].values
        }).set_index("Metric")

        st.bar_chart(comp_df)

#streamlit run "C:/Users/ecric/OneDrive/Desktop/mm_streamlit.py"

# Above is to keep handy the command to test the app.
