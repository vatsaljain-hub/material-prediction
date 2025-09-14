# Construction Project Schedule and Procurement Timeline Integration
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

class ConstructionScheduleManager:
    def __init__(self):
        self.project_start = datetime(2026, 2, 1)
        self.project_duration_weeks = 104  # 2 years for data center
        self.project_end = self.project_start + timedelta(weeks=self.project_duration_weeks)
        
    def create_construction_phases(self):
        """Create detailed construction phases for data center"""
        phases = [
            {
                'Phase': 'Site Preparation & Foundation',
                'Duration_Weeks': 12,
                'Start_Week': 0,
                'End_Week': 12,
                'Description': 'Site clearing, excavation, foundation work',
                'Critical_Path': True
            },
            {
                'Phase': 'Structural Construction',
                'Duration_Weeks': 20,
                'Start_Week': 8,
                'End_Week': 28,
                'Description': 'Concrete work, steel erection, building structure',
                'Critical_Path': True
            },
            {
                'Phase': 'MEP Rough-in',
                'Duration_Weeks': 16,
                'Start_Week': 20,
                'End_Week': 36,
                'Description': 'Mechanical, Electrical, Plumbing rough-in',
                'Critical_Path': True
            },
            {
                'Phase': 'HVAC Installation',
                'Duration_Weeks': 12,
                'Start_Week': 28,
                'End_Week': 40,
                'Description': 'HVAC systems, chillers, CRAC units',
                'Critical_Path': True
            },
            {
                'Phase': 'Electrical Systems',
                'Duration_Weeks': 16,
                'Start_Week': 32,
                'End_Week': 48,
                'Description': 'Power distribution, transformers, switchgear',
                'Critical_Path': True
            },
            {
                'Phase': 'Data Center Infrastructure',
                'Duration_Weeks': 20,
                'Start_Week': 40,
                'End_Week': 60,
                'Description': 'UPS, generators, fire suppression, monitoring',
                'Critical_Path': True
            },
            {
                'Phase': 'Interior Finishing',
                'Duration_Weeks': 12,
                'Start_Week': 48,
                'End_Week': 60,
                'Description': 'Flooring, ceilings, walls, doors',
                'Critical_Path': False
            },
            {
                'Phase': 'Testing & Commissioning',
                'Duration_Weeks': 8,
                'Start_Week': 56,
                'End_Week': 64,
                'Description': 'System testing, commissioning, validation',
                'Critical_Path': True
            },
            {
                'Phase': 'Final Handover',
                'Duration_Weeks': 4,
                'Start_Week': 60,
                'End_Week': 64,
                'Description': 'Final inspections, documentation, handover',
                'Critical_Path': True
            }
        ]
        
        # Convert to DataFrame and add dates
        df = pd.DataFrame(phases)
        df['Start_Date'] = df['Start_Week'].apply(lambda x: self.project_start + timedelta(weeks=x))
        df['End_Date'] = df['End_Week'].apply(lambda x: self.project_start + timedelta(weeks=x))
        
        return df
    
    def create_material_requirements_timeline(self, material_predictions):
        """Create material requirements timeline based on construction phases"""
        phases = self.create_construction_phases()
        
        # Map materials to construction phases
        material_phase_mapping = {
            'Cement': 'Site Preparation & Foundation',
            'Bricks': 'Structural Construction',
            'Steel Reinforcement': 'Structural Construction',
            'Medium Voltage Switchgear': 'Electrical Systems',
            'Transformers': 'Electrical Systems',
            'Chillers / CRAHs / CRACs': 'HVAC Installation',
            'HVAC Ductwork': 'HVAC Installation',
            'UPS Systems': 'Data Center Infrastructure',
            'Generator Sets': 'Data Center Infrastructure',
            'Fire Suppression Systems': 'Data Center Infrastructure'
        }
        
        material_timeline = []
        
        for _, material in material_predictions.iterrows():
            material_name = material['Material']
            quantity = material['Predicted_Quantity']
            phase_name = material_phase_mapping.get(material_name, 'Structural Construction')
            
            # Get phase information
            phase_info = phases[phases['Phase'] == phase_name].iloc[0]
            
            # Calculate delivery date (2 weeks before phase start for critical materials)
            delivery_weeks_before = 2 if phase_info['Critical_Path'] else 4
            delivery_date = phase_info['Start_Date'] - timedelta(weeks=delivery_weeks_before)
            
            # Calculate order date based on lead times
            lead_times = {
                'Medium Voltage Switchgear': 40,
                'Transformers': 50,
                'Chillers / CRAHs / CRACs': 30,
                'Cement': 2,
                'Bricks': 4,
                'Steel Reinforcement': 8,
                'UPS Systems': 20,
                'Generator Sets': 35,
                'Fire Suppression Systems': 15,
                'HVAC Ductwork': 6
            }
            
            lead_time_weeks = lead_times.get(material_name, 8)
            order_date = delivery_date - timedelta(weeks=lead_time_weeks)
            
            material_timeline.append({
                'Material': material_name,
                'Quantity': quantity,
                'UOM': material['UOM'],
                'Construction_Phase': phase_name,
                'Phase_Start': phase_info['Start_Date'].strftime('%Y-%m-%d'),
                'Phase_End': phase_info['End_Date'].strftime('%Y-%m-%d'),
                'Order_Date': order_date.strftime('%Y-%m-%d'),
                'Delivery_Date': delivery_date.strftime('%Y-%m-%d'),
                'Lead_Time_Weeks': lead_time_weeks,
                'Critical_Path': phase_info['Critical_Path'],
                'Estimated_Cost': material['Total_Estimated_Cost']
            })
        
        return pd.DataFrame(material_timeline)
    
    def create_procurement_schedule(self, material_timeline):
        """Create detailed procurement schedule with milestones"""
        procurement_schedule = []
        
        # Group by order date and create procurement milestones
        material_timeline['Order_Date'] = pd.to_datetime(material_timeline['Order_Date'])
        material_timeline = material_timeline.sort_values('Order_Date')
        
        current_date = self.project_start
        procurement_week = 0
        
        for _, material in material_timeline.iterrows():
            order_date = material['Order_Date']
            
            # Create procurement milestone
            milestone = {
                'Week': procurement_week,
                'Date': order_date.strftime('%Y-%m-%d'),
                'Milestone': f"Order {material['Material']}",
                'Material': material['Material'],
                'Quantity': material['Quantity'],
                'UOM': material['UOM'],
                'Delivery_Date': material['Delivery_Date'],
                'Lead_Time_Weeks': material['Lead_Time_Weeks'],
                'Estimated_Cost': material['Estimated_Cost'],
                'Critical_Path': material['Critical_Path'],
                'Status': 'Planned',
                'Responsible': 'Procurement Team',
                'Notes': f"Required for {material['Construction_Phase']}"
            }
            
            procurement_schedule.append(milestone)
            procurement_week += 1
        
        return pd.DataFrame(procurement_schedule)
    
    def create_resource_requirements(self, material_timeline):
        """Create resource requirements summary"""
        total_cost = material_timeline['Estimated_Cost'].sum()
        
        # Calculate monthly procurement costs
        material_timeline['Order_Date'] = pd.to_datetime(material_timeline['Order_Date'])
        material_timeline['Order_Month'] = material_timeline['Order_Date'].dt.to_period('M')
        
        monthly_costs = material_timeline.groupby('Order_Month')['Estimated_Cost'].sum().reset_index()
        monthly_costs['Order_Month'] = monthly_costs['Order_Month'].astype(str)
        
        resource_summary = {
            'Total_Project_Cost': total_cost,
            'Material_Cost': total_cost,
            'Material_Cost_Percentage': 35,  # Typical for data centers
            'Peak_Monthly_Procurement': monthly_costs['Estimated_Cost'].max(),
            'Average_Monthly_Procurement': monthly_costs['Estimated_Cost'].mean(),
            'Critical_Materials_Count': len(material_timeline[material_timeline['Critical_Path'] == True]),
            'Total_Materials_Count': len(material_timeline)
        }
        
        return resource_summary, monthly_costs
    
    def generate_integrated_schedule(self, material_predictions):
        """Generate complete integrated schedule"""
        print("Creating integrated construction and procurement schedule...")
        
        # Create construction phases
        construction_phases = self.create_construction_phases()
        
        # Create material timeline
        material_timeline = self.create_material_requirements_timeline(material_predictions)
        
        # Create procurement schedule
        procurement_schedule = self.create_procurement_schedule(material_timeline)
        
        # Create resource requirements
        resource_summary, monthly_costs = self.create_resource_requirements(material_timeline)
        
        return {
            'construction_phases': construction_phases,
            'material_timeline': material_timeline,
            'procurement_schedule': procurement_schedule,
            'resource_summary': resource_summary,
            'monthly_costs': monthly_costs
        }

def main():
    # Initialize schedule manager
    schedule_manager = ConstructionScheduleManager()
    
    # Load material predictions
    try:
        predictions = pd.read_csv('data_center_predictions.csv')
    except FileNotFoundError:
        print("Material predictions not found. Please run enhanced_model_training.py first.")
        return
    
    print("Creating integrated construction and procurement schedule...")
    print("="*60)
    
    # Generate integrated schedule
    schedule_data = schedule_manager.generate_integrated_schedule(predictions)
    
    # Save all components
    schedule_data['construction_phases'].to_csv('construction_phases.csv', index=False)
    schedule_data['material_timeline'].to_csv('material_timeline.csv', index=False)
    schedule_data['procurement_schedule'].to_csv('procurement_schedule.csv', index=False)
    schedule_data['monthly_costs'].to_csv('monthly_procurement_costs.csv', index=False)
    
    # Display summary
    print("\nCONSTRUCTION PHASES:")
    print(schedule_data['construction_phases'][['Phase', 'Start_Date', 'End_Date', 'Duration_Weeks', 'Critical_Path']].to_string(index=False))
    
    print("\nMATERIAL TIMELINE:")
    print(schedule_data['material_timeline'][['Material', 'Quantity', 'Order_Date', 'Delivery_Date', 'Construction_Phase']].to_string(index=False))
    
    print("\nPROCUREMENT SCHEDULE (First 10 items):")
    print(schedule_data['procurement_schedule'].head(10)[['Date', 'Milestone', 'Material', 'Quantity', 'Estimated_Cost']].to_string(index=False))
    
    print("\nRESOURCE SUMMARY:")
    summary = schedule_data['resource_summary']
    print(f"Total Project Cost: ₹{summary['Total_Project_Cost']:,.0f}")
    print(f"Material Cost: ₹{summary['Material_Cost']:,.0f}")
    print(f"Peak Monthly Procurement: ₹{summary['Peak_Monthly_Procurement']:,.0f}")
    print(f"Critical Materials: {summary['Critical_Materials_Count']}")
    print(f"Total Materials: {summary['Total_Materials_Count']}")
    
    print(f"\nAll schedules saved to CSV files.")

if __name__ == "__main__":
    main()
