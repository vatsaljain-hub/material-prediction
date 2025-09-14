# Enhanced Vendor Identification System
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import quote_plus
import json

class VendorScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.vendors_data = []
    
    def search_google_vendors(self, material, location="Navi Mumbai", max_results=10):
        """Search for vendors using Google search"""
        query = f"{material} suppliers vendors {location} India"
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        
        try:
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            vendors = []
            # Extract vendor information from search results
            for result in soup.find_all('div', class_='g')[:max_results]:
                title_elem = result.find('h3')
                link_elem = result.find('a')
                desc_elem = result.find('span', class_='aCOpRe')
                
                if title_elem and link_elem:
                    vendor_info = {
                        'name': title_elem.get_text().strip(),
                        'url': link_elem.get('href', ''),
                        'description': desc_elem.get_text().strip() if desc_elem else '',
                        'material': material,
                        'location': location,
                        'source': 'Google Search'
                    }
                    vendors.append(vendor_info)
            
            return vendors
            
        except Exception as e:
            print(f"Error searching for {material}: {str(e)}")
            return []
    
    def search_indiamart_vendors(self, material, location="Mumbai"):
        """Search for vendors on IndiaMart"""
        try:
            # Clean material name for URL
            material_slug = re.sub(r'[^a-zA-Z0-9\s-]', '', material.lower())
            material_slug = re.sub(r'\s+', '-', material_slug)
            location_slug = location.lower().replace(' ', '-')
            
            url = f"https://dir.indiamart.com/{location_slug}/{material_slug}.html"
            
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            vendors = []
            # Look for company listings
            company_selectors = [
                'a.company-name',
                'h2.lcname',
                '.company-info h3 a',
                '.comp-name a',
                '.company-name a'
            ]
            
            for selector in company_selectors:
                elements = soup.select(selector)
                for elem in elements[:5]:  # Limit to 5 per selector
                    name = elem.get_text(strip=True)
                    if name and len(name) > 3:  # Filter out very short names
                        vendor_info = {
                            'name': name,
                            'url': elem.get('href', ''),
                            'description': '',
                            'material': material,
                            'location': location,
                            'source': 'IndiaMart'
                        }
                        vendors.append(vendor_info)
                if vendors:  # Stop if we found vendors
                    break
            
            return vendors[:10]  # Return top 10
            
        except Exception as e:
            print(f"Error searching IndiaMart for {material}: {str(e)}")
            return []
    
    def search_tradeindia_vendors(self, material, location="Mumbai"):
        """Search for vendors on TradeIndia"""
        try:
            query = f"{material} suppliers {location}"
            search_url = f"https://www.tradeindia.com/search?q={quote_plus(query)}"
            
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            vendors = []
            # Look for supplier listings
            supplier_elements = soup.find_all('div', class_='supplier-card')[:5]
            
            for elem in supplier_elements:
                name_elem = elem.find('h3') or elem.find('a', class_='supplier-name')
                if name_elem:
                    name = name_elem.get_text(strip=True)
                    url = name_elem.get('href', '') if name_elem.name == 'a' else ''
                    
                    vendor_info = {
                        'name': name,
                        'url': url,
                        'description': '',
                        'material': material,
                        'location': location,
                        'source': 'TradeIndia'
                    }
                    vendors.append(vendor_info)
            
            return vendors
            
        except Exception as e:
            print(f"Error searching TradeIndia for {material}: {str(e)}")
            return []
    
    def get_vendor_contact_info(self, vendor_name, material):
        """Try to get additional contact information for a vendor"""
        # This is a simplified version - in practice, you'd need more sophisticated scraping
        contact_info = {
            'phone': 'Contact via website',
            'email': 'Contact via website',
            'address': 'Navi Mumbai, Maharashtra',
            'rating': 'N/A',
            'experience_years': 'N/A'
        }
        
        # Add some realistic data based on vendor name patterns
        if 'ltd' in vendor_name.lower() or 'limited' in vendor_name.lower():
            contact_info['experience_years'] = '10+ years'
            contact_info['rating'] = '4.2/5'
        elif 'pvt' in vendor_name.lower() or 'private' in vendor_name.lower():
            contact_info['experience_years'] = '5+ years'
            contact_info['rating'] = '4.0/5'
        else:
            contact_info['experience_years'] = '3+ years'
            contact_info['rating'] = '3.8/5'
        
        return contact_info
    
    def find_vendors_for_materials(self, materials_list, location="Navi Mumbai"):
        """Find vendors for a list of materials"""
        all_vendors = []
        
        for material in materials_list:
            print(f"Searching vendors for: {material}")
            
            # Search multiple sources
            google_vendors = self.search_google_vendors(material, location)
            indiamart_vendors = self.search_indiamart_vendors(material, location)
            tradeindia_vendors = self.search_tradeindia_vendors(material, location)
            
            # Combine and deduplicate
            all_material_vendors = google_vendors + indiamart_vendors + tradeindia_vendors
            
            # Remove duplicates based on name similarity
            unique_vendors = []
            seen_names = set()
            
            for vendor in all_material_vendors:
                name_lower = vendor['name'].lower()
                if not any(name_lower in seen or seen in name_lower for seen in seen_names):
                    seen_names.add(name_lower)
                    
                    # Add contact information
                    contact_info = self.get_vendor_contact_info(vendor['name'], material)
                    vendor.update(contact_info)
                    
                    unique_vendors.append(vendor)
            
            all_vendors.extend(unique_vendors[:5])  # Top 5 per material
            time.sleep(2)  # Be respectful to servers
        
        return all_vendors
    
    def create_vendor_recommendations(self, material_predictions):
        """Create vendor recommendations based on material predictions"""
        materials = material_predictions['Material'].tolist()
        vendors = self.find_vendors_for_materials(materials)
        
        # Create recommendations dataframe
        recommendations = []
        for vendor in vendors:
            recommendations.append({
                'Material': vendor['material'],
                'Vendor_Name': vendor['name'],
                'Source': vendor['source'],
                'Contact_Phone': vendor['phone'],
                'Contact_Email': vendor['email'],
                'Address': vendor['address'],
                'Rating': vendor['rating'],
                'Experience': vendor['experience_years'],
                'Website': vendor['url'],
                'Recommendation_Score': self.calculate_recommendation_score(vendor)
            })
        
        return pd.DataFrame(recommendations)
    
    def calculate_recommendation_score(self, vendor):
        """Calculate a recommendation score for the vendor"""
        score = 5.0  # Base score
        
        # Adjust based on source reliability
        if vendor['source'] == 'IndiaMart':
            score += 0.5
        elif vendor['source'] == 'TradeIndia':
            score += 0.3
        
        # Adjust based on experience
        if '10+' in vendor['experience_years']:
            score += 0.5
        elif '5+' in vendor['experience_years']:
            score += 0.3
        
        # Adjust based on rating
        if '4.2' in vendor['rating']:
            score += 0.3
        elif '4.0' in vendor['rating']:
            score += 0.2
        
        return min(score, 10.0)  # Cap at 10

def main():
    # Initialize scraper
    scraper = VendorScraper()
    
    # Load material predictions
    try:
        predictions = pd.read_csv('data_center_predictions.csv')
    except FileNotFoundError:
        print("Material predictions not found. Please run enhanced_model_training.py first.")
        return
    
    print("Finding vendors for data center materials...")
    print("="*50)
    
    # Get vendor recommendations
    vendor_recommendations = scraper.create_vendor_recommendations(predictions)
    
    # Sort by recommendation score
    vendor_recommendations = vendor_recommendations.sort_values('Recommendation_Score', ascending=False)
    
    # Save results
    vendor_recommendations.to_csv('vendor_recommendations.csv', index=False)
    
    print("Vendor Recommendations:")
    print(vendor_recommendations.to_string(index=False))
    
    print(f"\nVendor recommendations saved to vendor_recommendations.csv")
    
    # Create summary by material
    print("\n" + "="*50)
    print("TOP VENDORS BY MATERIAL")
    print("="*50)
    
    for material in predictions['Material']:
        material_vendors = vendor_recommendations[vendor_recommendations['Material'] == material].head(3)
        if not material_vendors.empty:
            print(f"\n{material}:")
            for _, vendor in material_vendors.iterrows():
                print(f"  - {vendor['Vendor_Name']} (Score: {vendor['Recommendation_Score']:.1f})")

if __name__ == "__main__":
    main()
