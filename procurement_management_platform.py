# Comprehensive Procurement Management Platform
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import json

class ProcurementManagementPlatform:
    def __init__(self):
        self.platform_data = {}
        self.load_platform_data()
    
    def load_platform_data(self):
        """Load all platform data"""
        try:
            self.platform_data = {
                'predictions': pd.read_csv('data_center_predictions.csv'),
                'vendors': pd.read_csv('vendor_recommendations.csv'),
                'schedule': pd.read_csv('procurement_schedule.csv'),
                'construction_phases': pd.read_csv('construction_phases.csv'),
                'material_timeline': pd.read_csv('material_timeline.csv')
            }
        except FileNotFoundError as e:
            st.error(f"Required data file not found: {e}")
    
    def render_dashboard_overview(self):
        """Render main dashboard overview"""
        st.header("🎯 Procurement Management Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_materials = len(self.platform_data.get('predictions', []))
            st.metric("Total Materials", total_materials, "10 types")
        
        with col2:
            total_cost = self.platform_data.get('predictions', pd.DataFrame())['Total_Estimated_Cost'].sum()
            st.metric("Total Budget", f"₹{total_cost:,.0f}", "Material cost")
        
        with col3:
            vendor_count = len(self.platform_data.get('vendors', []))
            st.metric("Vendor Pool", vendor_count, "Verified suppliers")
        
        with col4:
            critical_items = len(self.platform_data.get('schedule', pd.DataFrame())[
                self.platform_data.get('schedule', pd.DataFrame())['Critical_Path'] == True
            ])
            st.metric("Critical Items", critical_items, "On critical path")
        
        # Risk assessment
        self.render_risk_assessment()
    
    def render_risk_assessment(self):
        """Render risk assessment section"""
        st.subheader("⚠️ Risk Assessment")
        
        if 'schedule' not in self.platform_data:
            st.warning("Schedule data not available for risk assessment")
            return
        
        schedule = self.platform_data['schedule']
        schedule['Date'] = pd.to_datetime(schedule['Date'])
        schedule['Delivery_Date'] = pd.to_datetime(schedule['Delivery_Date'])
        
        # Calculate risks
        current_date = datetime.now()
        upcoming_orders = schedule[schedule['Date'] <= current_date + timedelta(days=30)]
        overdue_orders = schedule[schedule['Date'] < current_date]
        critical_upcoming = upcoming_orders[upcoming_orders['Critical_Path'] == True]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Upcoming Orders (30 days)", len(upcoming_orders), "Action required")
        
        with col2:
            st.metric("Overdue Orders", len(overdue_orders), "High risk" if len(overdue_orders) > 0 else "On track")
        
        with col3:
            st.metric("Critical Upcoming", len(critical_upcoming), "Priority items")
        
        # Risk visualization
        if len(upcoming_orders) > 0:
            fig_risk = px.bar(
                upcoming_orders,
                x='Material',
                y='Estimated_Cost',
                color='Critical_Path',
                title="Upcoming Orders Risk Analysis",
                labels={'Estimated_Cost': 'Cost (₹)', 'Material': 'Material Type'}
            )
            fig_risk.update_xaxes(tickangle=45)
            st.plotly_chart(fig_risk, use_container_width=True)
    
    def render_vendor_management(self):
        """Render vendor management section"""
        st.header("🏢 Vendor Management")
        
        if 'vendors' not in self.platform_data:
            st.error("Vendor data not available")
            return
        
        vendors = self.platform_data['vendors']
        
        # Vendor analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Vendor distribution by source
            source_counts = vendors['Source'].value_counts()
            fig_source = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Vendor Distribution by Source"
            )
            st.plotly_chart(fig_source, use_container_width=True)
        
        with col2:
            # Top vendors by recommendation score
            top_vendors = vendors.nlargest(10, 'Recommendation_Score')
            fig_top = px.bar(
                top_vendors,
                x='Recommendation_Score',
                y='Vendor_Name',
                orientation='h',
                title="Top 10 Vendors by Score"
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        # Vendor selection interface
        st.subheader("Vendor Selection Interface")
        
        # Material selection
        material = st.selectbox("Select Material", vendors['Material'].unique())
        
        # Filter vendors for selected material
        material_vendors = vendors[vendors['Material'] == material].sort_values('Recommendation_Score', ascending=False)
        
        if not material_vendors.empty:
            st.write(f"**Best vendors for {material}:**")
            
            for idx, vendor in material_vendors.iterrows():
                with st.expander(f"🏢 {vendor['Vendor_Name']} (Score: {vendor['Recommendation_Score']:.1f})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Source:** {vendor['Source']}")
                        st.write(f"**Rating:** {vendor['Rating']}")
                        st.write(f"**Experience:** {vendor['Experience']}")
                        st.write(f"**Phone:** {vendor['Contact_Phone']}")
                    
                    with col2:
                        st.write(f"**Email:** {vendor['Contact_Email']}")
                        st.write(f"**Address:** {vendor['Address']}")
                        if vendor['Website']:
                            st.write(f"**Website:** {vendor['Website']}")
                    
                    # Selection buttons
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    with col_btn1:
                        if st.button(f"Select Primary", key=f"primary_{idx}"):
                            st.success(f"Selected {vendor['Vendor_Name']} as primary vendor")
                    with col_btn2:
                        if st.button(f"Select Backup", key=f"backup_{idx}"):
                            st.info(f"Selected {vendor['Vendor_Name']} as backup vendor")
                    with col_btn3:
                        if st.button(f"Request Quote", key=f"quote_{idx}"):
                            st.info(f"Quote request sent to {vendor['Vendor_Name']}")
    
    def render_procurement_workflow(self):
        """Render procurement workflow management"""
        st.header("📋 Procurement Workflow")
        
        # Workflow stages
        workflow_stages = [
            {"stage": "Planning", "status": "Completed", "description": "Material requirements identified"},
            {"stage": "Vendor Selection", "status": "In Progress", "description": "Evaluating vendor options"},
            {"stage": "RFQ Process", "status": "Pending", "description": "Request for quotes"},
            {"stage": "Negotiation", "status": "Pending", "description": "Price and terms negotiation"},
            {"stage": "Contract Award", "status": "Pending", "description": "Final vendor selection"},
            {"stage": "Order Placement", "status": "Pending", "description": "Purchase order creation"},
            {"stage": "Delivery Tracking", "status": "Pending", "description": "Monitor delivery status"},
            {"stage": "Quality Check", "status": "Pending", "description": "Incoming material inspection"},
            {"stage": "Payment", "status": "Pending", "description": "Invoice processing"}
        ]
        
        # Display workflow
        for i, stage in enumerate(workflow_stages):
            col1, col2, col3 = st.columns([1, 3, 2])
            
            with col1:
                if stage["status"] == "Completed":
                    st.success("✅")
                elif stage["status"] == "In Progress":
                    st.warning("🔄")
                else:
                    st.info("⏳")
            
            with col2:
                st.write(f"**{stage['stage']}**")
                st.write(stage['description'])
            
            with col3:
                st.write(f"Status: {stage['status']}")
        
        # Procurement calendar
        st.subheader("📅 Procurement Calendar")
        
        if 'schedule' in self.platform_data:
            schedule = self.platform_data['schedule']
            schedule['Date'] = pd.to_datetime(schedule['Date'])
            
            # Create calendar view
            calendar_data = []
            for _, row in schedule.iterrows():
                calendar_data.append({
                    'Date': row['Date'],
                    'Material': row['Material'],
                    'Milestone': row['Milestone'],
                    'Cost': row['Estimated_Cost'],
                    'Critical': row['Critical_Path']
                })
            
            calendar_df = pd.DataFrame(calendar_data)
            
            # Monthly view
            monthly_view = calendar_df.groupby(calendar_df['Date'].dt.to_period('M')).agg({
                'Cost': 'sum',
                'Material': 'count'
            }).reset_index()
            monthly_view['Date'] = monthly_view['Date'].astype(str)
            
            fig_calendar = px.bar(
                monthly_view,
                x='Date',
                y='Cost',
                title="Monthly Procurement Schedule",
                labels={'Cost': 'Cost (₹)', 'Date': 'Month', 'Material': 'Number of Items'}
            )
            st.plotly_chart(fig_calendar, use_container_width=True)
    
    def render_cost_management(self):
        """Render cost management section"""
        st.header("💰 Cost Management")
        
        if 'predictions' not in self.platform_data:
            st.error("Cost data not available")
            return
        
        predictions = self.platform_data['predictions']
        
        # Cost breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost distribution
            fig_cost = px.pie(
                predictions,
                values='Total_Estimated_Cost',
                names='Material',
                title="Material Cost Distribution"
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col2:
            # Cost vs quantity analysis
            fig_analysis = px.scatter(
                predictions,
                x='Predicted_Quantity',
                y='Total_Estimated_Cost',
                size='Estimated_Cost_Per_Unit',
                hover_data=['Material', 'UOM'],
                title="Cost vs Quantity Analysis"
            )
            st.plotly_chart(fig_analysis, use_container_width=True)
        
        # Budget tracking
        st.subheader("📊 Budget Tracking")
        
        total_budget = predictions['Total_Estimated_Cost'].sum()
        contingency = total_budget * 0.1  # 10% contingency
        total_with_contingency = total_budget + contingency
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Base Budget", f"₹{total_budget:,.0f}")
        
        with col2:
            st.metric("Contingency (10%)", f"₹{contingency:,.0f}")
        
        with col3:
            st.metric("Total Budget", f"₹{total_with_contingency:,.0f}")
        
        # Cost optimization suggestions
        st.subheader("💡 Cost Optimization Suggestions")
        
        suggestions = [
            "Consider bulk purchasing for high-volume materials like cement and bricks",
            "Negotiate long-term contracts with reliable vendors for critical equipment",
            "Explore alternative suppliers for non-critical materials to reduce costs",
            "Implement just-in-time delivery for materials with short lead times",
            "Consider leasing options for expensive equipment like generators"
        ]
        
        for i, suggestion in enumerate(suggestions, 1):
            st.write(f"{i}. {suggestion}")
    
    def render_quality_management(self):
        """Render quality management section"""
        st.header("🔍 Quality Management")
        
        # Quality standards
        quality_standards = {
            'Cement': ['IS 12269:2013', 'IS 1489:2015'],
            'Steel Reinforcement': ['IS 1786:2008', 'IS 432:1982'],
            'Transformers': ['IS 2026:2011', 'IEC 60076'],
            'UPS Systems': ['IS 16242:2014', 'IEC 62040'],
            'Fire Suppression Systems': ['IS 15105:2002', 'NFPA 2001']
        }
        
        st.subheader("📋 Quality Standards by Material")
        
        for material, standards in quality_standards.items():
            with st.expander(f"📄 {material}"):
                st.write("**Applicable Standards:**")
                for standard in standards:
                    st.write(f"• {standard}")
        
        # Quality checklist
        st.subheader("✅ Quality Checklist")
        
        checklist_items = [
            "Material specifications match project requirements",
            "Vendor certifications are valid and up-to-date",
            "Quality test certificates are available",
            "Delivery schedule aligns with construction timeline",
            "Packaging and handling requirements are met",
            "Warranty terms are clearly defined",
            "Return and replacement policies are documented"
        ]
        
        for item in checklist_items:
            st.checkbox(item, key=f"quality_{item}")
    
    def render_reporting_analytics(self):
        """Render reporting and analytics section"""
        st.header("📈 Reporting & Analytics")
        
        # Generate reports
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Material Report"):
                st.success("Material report generated successfully!")
        
        with col2:
            if st.button("🏢 Generate Vendor Report"):
                st.success("Vendor report generated successfully!")
        
        with col3:
            if st.button("📅 Generate Schedule Report"):
                st.success("Schedule report generated successfully!")
        
        # Analytics dashboard
        st.subheader("📊 Analytics Dashboard")
        
        if 'schedule' in self.platform_data:
            schedule = self.platform_data['schedule']
            
            # Procurement timeline analysis
            schedule['Date'] = pd.to_datetime(schedule['Date'])
            schedule['Month'] = schedule['Date'].dt.to_period('M')
            
            monthly_analysis = schedule.groupby('Month').agg({
                'Estimated_Cost': 'sum',
                'Material': 'count',
                'Critical_Path': 'sum'
            }).reset_index()
            
            monthly_analysis['Month'] = monthly_analysis['Month'].astype(str)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Costs', 'Order Count', 'Critical Items', 'Cost Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "pie"}]]
            )
            
            # Monthly costs
            fig.add_trace(
                go.Bar(x=monthly_analysis['Month'], y=monthly_analysis['Estimated_Cost'], name='Cost'),
                row=1, col=1
            )
            
            # Order count
            fig.add_trace(
                go.Bar(x=monthly_analysis['Month'], y=monthly_analysis['Material'], name='Orders'),
                row=1, col=2
            )
            
            # Critical items
            fig.add_trace(
                go.Bar(x=monthly_analysis['Month'], y=monthly_analysis['Critical_Path'], name='Critical'),
                row=2, col=1
            )
            
            # Cost distribution pie
            if 'predictions' in self.platform_data:
                predictions = self.platform_data['predictions']
                fig.add_trace(
                    go.Pie(labels=predictions['Material'], values=predictions['Total_Estimated_Cost'], name='Cost Dist'),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=False, title_text="Procurement Analytics Dashboard")
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Initialize platform
    platform = ProcurementManagementPlatform()
    
    # Page configuration
    st.set_page_config(
        page_title="Procurement Management Platform",
        page_icon="🏗️",
        layout="wide"
    )
    
    # Main title
    st.title("🏗️ Data Center Procurement Management Platform")
    st.markdown("**Comprehensive procurement management for 25MW Data Center project in Navi Mumbai**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        ["Dashboard Overview", "Vendor Management", "Procurement Workflow", 
         "Cost Management", "Quality Management", "Reporting & Analytics"]
    )
    
    # Render selected page
    if page == "Dashboard Overview":
        platform.render_dashboard_overview()
    elif page == "Vendor Management":
        platform.render_vendor_management()
    elif page == "Procurement Workflow":
        platform.render_procurement_workflow()
    elif page == "Cost Management":
        platform.render_cost_management()
    elif page == "Quality Management":
        platform.render_quality_management()
    elif page == "Reporting & Analytics":
        platform.render_reporting_analytics()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Procurement Management Platform v2.0**")
    st.sidebar.markdown("Data Center Project | Navi Mumbai")

if __name__ == "__main__":
    main()
