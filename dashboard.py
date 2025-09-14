# Enhanced Material Forecasting Dashboard with Advanced Features
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Data Center Material Forecasting Platform",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class MaterialForecastingDashboard:
    def __init__(self):
        self.model_file = "enhanced_material_model.pkl"
        self.predictions_file = "data_center_predictions.csv"
        self.vendor_file = "vendor_recommendations.csv"
        self.schedule_file = "procurement_schedule.csv"
        
    def load_data(self):
        """Load all required data files"""
        data = {}
        
        # Load model
        if os.path.exists(self.model_file):
            try:
                model_data = joblib.load(self.model_file)
                data['model'] = model_data.get('model')
                data['feature_importance'] = model_data.get('feature_importance')
                data['performance'] = model_data.get('performance', {})
            except Exception as e:
                print(f"Error loading model: {e}")
                data['model'] = None
                data['feature_importance'] = None
                data['performance'] = {}
        
        # Load predictions
        if os.path.exists(self.predictions_file):
            try:
                data['predictions'] = pd.read_csv(self.predictions_file)
            except Exception as e:
                print(f"Error loading predictions: {e}")
                data['predictions'] = None
        
        # Load vendor recommendations
        if os.path.exists(self.vendor_file):
            try:
                data['vendors'] = pd.read_csv(self.vendor_file)
            except Exception as e:
                print(f"Error loading vendors: {e}")
                data['vendors'] = None
        
        # Load procurement schedule
        if os.path.exists(self.schedule_file):
            try:
                data['schedule'] = pd.read_csv(self.schedule_file)
            except Exception as e:
                print(f"Error loading schedule: {e}")
                data['schedule'] = None
        
        return data
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🏗️ Data Center Material Forecasting Platform</h1>', unsafe_allow_html=True)
        st.markdown("**Project:** 25MW Data Center | **Location:** Navi Mumbai | **Budget:** ₹1,875 Cr")
    
    def render_project_overview(self, data):
        """Render project overview metrics"""
        st.header("📊 Project Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Materials",
                value=len(data.get('predictions', [])),
                delta="10 materials"
            )
        
        with col2:
            total_cost = data.get('predictions', pd.DataFrame())['Total_Estimated_Cost'].sum() if 'predictions' in data else 0
            st.metric(
                label="Total Material Cost",
                value=f"₹{total_cost:,.0f}",
                delta="Estimated"
            )
        
        with col3:
            vendor_count = len(data.get('vendors', [])) if 'vendors' in data else 0
            st.metric(
                label="Vendor Recommendations",
                value=vendor_count,
                delta="Verified suppliers"
            )
        
        with col4:
            schedule_items = len(data.get('schedule', [])) if 'schedule' in data else 0
            st.metric(
                label="Procurement Milestones",
                value=schedule_items,
                delta="Scheduled orders"
            )
    
    def render_material_predictions(self, data):
        """Render material predictions with visualizations"""
        st.header("🔮 Material Predictions")
        
        if 'predictions' not in data:
            st.error("Material predictions not found. Please run the model training first.")
            return
        
        predictions = data['predictions']
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Summary Table", "📊 Cost Analysis", "📈 Quantity Distribution"])
        
        with tab1:
            st.subheader("Predicted Material Requirements")
            
            # Display predictions table
            display_df = predictions[['Material', 'UOM', 'Predicted_Quantity', 'Estimated_Cost_Per_Unit', 'Total_Estimated_Cost']].copy()
            display_df['Total_Estimated_Cost'] = display_df['Total_Estimated_Cost'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Estimated_Cost_Per_Unit'] = display_df['Estimated_Cost_Per_Unit'].apply(lambda x: f"₹{x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True)
        
        with tab2:
            st.subheader("Cost Analysis")
            
            # Cost breakdown pie chart
            fig_pie = px.pie(
                predictions, 
                values='Total_Estimated_Cost', 
                names='Material',
                title="Material Cost Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Cost vs Quantity scatter
            fig_scatter = px.scatter(
                predictions,
                x='Predicted_Quantity',
                y='Total_Estimated_Cost',
                size='Estimated_Cost_Per_Unit',
                hover_data=['Material', 'UOM'],
                title="Cost vs Quantity Analysis",
                labels={'Predicted_Quantity': 'Quantity', 'Total_Estimated_Cost': 'Total Cost (₹)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            st.subheader("Quantity Distribution")
            
            # Bar chart of quantities
            fig_bar = px.bar(
                predictions,
                x='Material',
                y='Predicted_Quantity',
                title="Predicted Material Quantities",
                labels={'Predicted_Quantity': 'Quantity', 'Material': 'Material Type'}
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def render_vendor_recommendations(self, data):
        """Render vendor recommendations"""
        st.header("🏢 Vendor Recommendations")
        
        if 'vendors' not in data:
            st.warning("Vendor recommendations not found. Please run the vendor scraper first.")
            return
        
        vendors = data['vendors']
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            material_filter = st.selectbox("Filter by Material", ["All"] + list(vendors['Material'].unique()))
        with col2:
            min_score = st.slider("Minimum Recommendation Score", 0.0, 10.0, 5.0)
        
        # Apply filters
        filtered_vendors = vendors.copy()
        if material_filter != "All":
            filtered_vendors = filtered_vendors[filtered_vendors['Material'] == material_filter]
        filtered_vendors = filtered_vendors[filtered_vendors['Recommendation_Score'] >= min_score]
        
        # Display vendor recommendations
        st.subheader(f"Top Vendor Recommendations (Score ≥ {min_score})")
        
        # Create vendor cards
        for _, vendor in filtered_vendors.head(10).iterrows():
            with st.expander(f"🏢 {vendor['Vendor_Name']} - {vendor['Material']} (Score: {vendor['Recommendation_Score']:.1f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Source:** {vendor['Source']}")
                    st.write(f"**Rating:** {vendor['Rating']}")
                    st.write(f"**Experience:** {vendor['Experience']}")
                
                with col2:
                    st.write(f"**Phone:** {vendor['Contact_Phone']}")
                    st.write(f"**Email:** {vendor['Contact_Email']}")
                    st.write(f"**Address:** {vendor['Address']}")
                
                if vendor['Website']:
                    st.write(f"**Website:** {vendor['Website']}")
    
    def render_procurement_schedule(self, data):
        """Render procurement schedule and timeline"""
        st.header("📅 Procurement Schedule")
        
        if 'schedule' not in data:
            st.warning("Procurement schedule not found. Please run the schedule generator first.")
            return
        
        schedule = data['schedule']
        
        # Convert dates
        schedule['Date'] = pd.to_datetime(schedule['Date'])
        schedule['Delivery_Date'] = pd.to_datetime(schedule['Delivery_Date'])
        
        # Timeline visualization
        st.subheader("Procurement Timeline")
        
        # Create Gantt chart
        fig_gantt = px.timeline(
            schedule,
            x_start='Date',
            x_end='Delivery_Date',
            y='Material',
            color='Critical_Path',
            title="Procurement Timeline",
            labels={'Date': 'Order Date', 'Delivery_Date': 'Delivery Date'}
        )
        fig_gantt.update_layout(height=600)
        st.plotly_chart(fig_gantt, use_container_width=True)
        
        # Monthly procurement costs
        schedule['Order_Month'] = schedule['Date'].dt.to_period('M')
        monthly_costs = schedule.groupby('Order_Month')['Estimated_Cost'].sum().reset_index()
        monthly_costs['Order_Month'] = monthly_costs['Order_Month'].astype(str)
        
        fig_monthly = px.bar(
            monthly_costs,
            x='Order_Month',
            y='Estimated_Cost',
            title="Monthly Procurement Costs",
            labels={'Estimated_Cost': 'Cost (₹)', 'Order_Month': 'Month'}
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Critical path items
        st.subheader("Critical Path Items")
        critical_items = schedule[schedule['Critical_Path'] == True]
        st.dataframe(critical_items[['Date', 'Milestone', 'Material', 'Quantity', 'Estimated_Cost']], use_container_width=True)
    
    def render_chatbot(self, data):
        """Render enhanced chatbot interface"""
        st.header("🤖 AI Assistant")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about materials, vendors, or procurement schedule..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                response = self.generate_chatbot_response(prompt, data)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    def generate_chatbot_response(self, prompt, data):
        """Generate chatbot response based on prompt and data"""
        prompt_lower = prompt.lower()
        
        # Material-related queries
        if any(word in prompt_lower for word in ['material', 'quantity', 'prediction']):
            if 'predictions' in data:
                materials = data['predictions']['Material'].tolist()
                return f"I can help you with material predictions. We have predictions for: {', '.join(materials[:5])}. What specific material would you like to know about?"
            else:
                return "Material predictions are not available. Please run the model training first."
        
        # Vendor-related queries
        elif any(word in prompt_lower for word in ['vendor', 'supplier', 'supply']):
            if 'vendors' in data:
                vendor_count = len(data['vendors'])
                return f"I have {vendor_count} vendor recommendations available. You can filter them by material type or recommendation score in the Vendor Recommendations section."
            else:
                return "Vendor recommendations are not available. Please run the vendor scraper first."
        
        # Schedule-related queries
        elif any(word in prompt_lower for word in ['schedule', 'timeline', 'order', 'delivery']):
            if 'schedule' in data:
                critical_items = len(data['schedule'][data['schedule']['Critical_Path'] == True])
                return f"The procurement schedule includes {len(data['schedule'])} milestones with {critical_items} critical path items. Check the Procurement Schedule section for detailed timeline."
            else:
                return "Procurement schedule is not available. Please run the schedule generator first."
        
        # Cost-related queries
        elif any(word in prompt_lower for word in ['cost', 'budget', 'price']):
            if 'predictions' in data:
                total_cost = data['predictions']['Total_Estimated_Cost'].sum()
                return f"The total estimated material cost is ₹{total_cost:,.0f}. You can see the cost breakdown in the Material Predictions section."
            else:
                return "Cost information is not available. Please run the model training first."
        
        # General help
        else:
            return """I can help you with:
            - Material predictions and quantities
            - Vendor recommendations and contact details
            - Procurement schedule and timelines
            - Cost analysis and budgeting
            
            
            What would you like to know more about?"""

def main():
    # Initialize dashboard
    dashboard = MaterialForecastingDashboard()
    
    # Load data
    data = dashboard.load_data()
    
    # Render header
    dashboard.render_header()
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Project Overview", "Material Predictions", "Vendor Recommendations", 
         "Procurement Schedule", "AI Assistant"]
    )
    
    # Render selected page
    if page == "Project Overview":
        dashboard.render_project_overview(data)
    elif page == "Material Predictions":
        dashboard.render_material_predictions(data)
    elif page == "Vendor Recommendations":
        dashboard.render_vendor_recommendations(data)
    elif page == "Procurement Schedule":
        dashboard.render_procurement_schedule(data)
    elif page == "AI Assistant":
        dashboard.render_chatbot(data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Center Material Forecasting Platform**")
    st.sidebar.markdown("Version 2.0 | Enhanced Features")

if __name__ == "__main__":
    main()
