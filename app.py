import streamlit as st
import duckdb
import time
import plotly.graph_objects as go
import plotly.express as px

# UI Configuration
st.set_page_config(page_title="Live Sentiment Engine", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    header {visibility: hidden;}
    h1, h2, h3 { color: #58a6ff; }
    [data-testid="stMetricValue"] { color: #2ea043; }
</style>
""", unsafe_allow_html=True)

# 1. Sidebar Controls & Filters
st.sidebar.title("⚙️ Dashboard Controls")
is_paused = st.sidebar.toggle("Pause Live Feed", value=False)

st.sidebar.markdown("### Interactive Filters")
platforms = st.sidebar.multiselect("Platforms", ["X", "Reddit", "LinkedIn", "Instagram"], default=["X", "Reddit", "LinkedIn", "Instagram"])
sentiments = st.sidebar.multiselect("Sentiments", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])

st.title("Live Social Media Sentiment Engine")

# Format lists for the SQL IN clause
plat_str = ", ".join([f"'{p}'" for p in platforms]) if platforms else "'None'"
sent_str = ", ".join([f"'{s}'" for s in sentiments]) if sentiments else "'None'"

placeholder = st.empty()

while True:
    if not is_paused:
        try:
            # Query: Main Metrics
            query_stats = f"""
            SELECT 
                COUNT(*) as total,
                AVG(polarity) as avg_polarity,
                SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as pos_count,
                SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as neg_count,
                SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) as neu_count
            FROM read_json_auto('social_stream.jsonl')
            WHERE platform IN ({plat_str}) AND sentiment IN ({sent_str})
            """
            
            # Query: Live Feed
            query_feed = f"""
            SELECT timestamp, username, platform, text, polarity, sentiment
            FROM read_json_auto('social_stream.jsonl')
            WHERE platform IN ({plat_str}) AND sentiment IN ({sent_str})
            ORDER BY timestamp DESC
            LIMIT 15
            """

            # Query: Rolling Trendline (Grouped by minute)
            query_trend = f"""
            SELECT 
                time_bucket(INTERVAL '1 minute', CAST(timestamp AS TIMESTAMP)) as time_window,
                AVG(polarity) as avg_polarity
            FROM read_json_auto('social_stream.jsonl')
            WHERE platform IN ({plat_str}) AND sentiment IN ({sent_str})
            GROUP BY time_window
            ORDER BY time_window ASC
            """

            # Query: Platform Breakdown
            query_platform = f"""
            SELECT platform, COUNT(*) as count, AVG(polarity) as avg_polarity
            FROM read_json_auto('social_stream.jsonl')
            WHERE platform IN ({plat_str}) AND sentiment IN ({sent_str})
            GROUP BY platform
            ORDER BY count DESC
            """

            stats_df = duckdb.query(query_stats).df()
            feed_df = duckdb.query(query_feed).df()
            trend_df = duckdb.query(query_trend).df()
            plat_df = duckdb.query(query_platform).df()

            # 2. Critical Threat Toasts
            if not feed_df.empty:
                latest_post = feed_df.iloc[0]
                if latest_post['polarity'] < -0.8:
                    st.toast(f"Critical Alert: Highly negative post detected on {latest_post['platform']}", icon="🚨")

            if len(stats_df) > 0 and stats_df['total'][0] > 0:
                total = int(stats_df['total'][0])
                avg_pol = float(stats_df['avg_polarity'][0])
                pos, neg, neu = int(stats_df['pos_count'][0]), int(stats_df['neg_count'][0]), int(stats_df['neu_count'][0])
            else:
                total = avg_pol = pos = neg = neu = 0

            with placeholder.container():
                # Top Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Posts Filtered", total)
                col2.metric("Positive Mentions", pos)
                col3.metric("Neutral Mentions", neu)
                col4.metric("Negative Mentions", neg)
                st.markdown("---")

                # Charts Row
                chart_col1, chart_col2 = st.columns([2, 1])
                with chart_col1:
                    st.subheader("Rolling Sentiment Trend (Per Minute)")
                    if not trend_df.empty:
                        fig_trend = px.line(trend_df, x="time_window", y="avg_polarity", markers=True)
                        fig_trend.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "#c9d1d9"}, xaxis_title="", yaxis_title="Average Polarity",
                            margin=dict(l=0, r=0, t=10, b=0)
                        )
                        fig_trend.update_traces(line_color='#58a6ff')
                        st.plotly_chart(fig_trend, use_container_width=True)

                with chart_col2:
                    st.subheader("Platform Volume")
                    if not plat_df.empty:
                        fig_plat = px.bar(plat_df, x="count", y="platform", orientation='h', color="avg_polarity", color_continuous_scale="RdYlGn")
                        fig_plat.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "#c9d1d9"}, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False
                        )
                        st.plotly_chart(fig_plat, use_container_width=True)
                st.markdown("---")

                # Feed & Gauge Row
                feed_col, gauge_col = st.columns([2, 1])
                with gauge_col:
                    st.subheader("Filtered Health Score")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_pol,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#c9d1d9"},
                            'bar': {'color': "#58a6ff"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'steps': [
                                {'range': [-1, -0.2], 'color': "#ff7b72"},
                                {'range': [-0.2, 0.2], 'color': "#30363d"},
                                {'range': [0.2, 1], 'color': "#2ea043"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#c9d1d9"}, height=280, margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # 3. Data Export Control
                    csv_export = duckdb.query(f"SELECT * FROM read_json_auto('social_stream.jsonl') WHERE platform IN ({plat_str}) AND sentiment IN ({sent_str})").df().to_csv(index=False)
                    st.download_button(label="📥 Export Filtered View to CSV", data=csv_export, file_name="sentiment_export.csv", mime="text/csv", use_container_width=True)

                with feed_col:
                    st.subheader("Live Feed & Alerts")
                    for _, row in feed_df.iterrows():
                        avatar = "🚨" if row['sentiment'] == 'Negative' else "✅" if row['sentiment'] == 'Positive' else "💬"
                        with st.chat_message(row['username'], avatar=avatar):
                            st.markdown(f"**{row['platform']}** | Polarity: `{row['polarity']:.2f}`")
                            st.write(row['text'])
                            
        except Exception:
            pass

    # Controls the loop refresh rate
    time.sleep(1.5)