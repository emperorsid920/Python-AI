import requests
from bs4 import BeautifulSoup
import csv
import time
import random
import os
import json
import re
import datetime
import argparse
from typing import List, Dict, Any, Optional

try:
    import anthropic
except ImportError:
    print("Anthropic package not installed. Install it with: pip install anthropic")
    anthropic = None


class YCScraper:
    def __init__(self, output_file: str = "yc_startups.csv", claude_api_key: Optional[str] = None):
        self.base_url = "https://www.ycombinator.com/companies"
        self.output_file = output_file
        self.claude_api_key = claude_api_key
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }

        # Initialize Claude client if API key is provided
        self.claude_client = None
        if anthropic and self.claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
                print("Claude API client initialized successfully")
            except Exception as e:
                print(f"Error initializing Claude API client: {e}")

    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            # Log the response status for debugging
            print(f"Successfully fetched {url} - Status code: {response.status_code}")

            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def get_startup_links(self, num_startups: int = 10) -> List[str]:
        """Get links to individual startup pages."""
        # Using a curated list of known YC companies to avoid the founders directory issue
        print("Using a curated list of well-known YC companies")
        known_companies = [
            "https://www.ycombinator.com/companies/airbnb",
            "https://www.ycombinator.com/companies/stripe",
            "https://www.ycombinator.com/companies/dropbox",
            "https://www.ycombinator.com/companies/instacart",
            "https://www.ycombinator.com/companies/doordash",
            "https://www.ycombinator.com/companies/coinbase",
            "https://www.ycombinator.com/companies/gitlab",
            "https://www.ycombinator.com/companies/reddit",
            "https://www.ycombinator.com/companies/twitch",
            "https://www.ycombinator.com/companies/zapier",
            "https://www.ycombinator.com/companies/amplitude",
            "https://www.ycombinator.com/companies/gumroad",
            "https://www.ycombinator.com/companies/weebly",
            "https://www.ycombinator.com/companies/openai",
            "https://www.ycombinator.com/companies/gusto",
            "https://www.ycombinator.com/companies/plaid",
            "https://www.ycombinator.com/companies/cruise",
            "https://www.ycombinator.com/companies/rappi",
            "https://www.ycombinator.com/companies/heroku",
            "https://www.ycombinator.com/companies/brex",
            "https://www.ycombinator.com/companies/segment",
            "https://www.ycombinator.com/companies/razorpay",
            "https://www.ycombinator.com/companies/meesho",
            "https://www.ycombinator.com/companies/clearbit",
            "https://www.ycombinator.com/companies/monzo",
            "https://www.ycombinator.com/companies/rappi",
            "https://www.ycombinator.com/companies/notion",
            "https://www.ycombinator.com/companies/faire",
            "https://www.ycombinator.com/companies/rippling",
            "https://www.ycombinator.com/companies/flexport"
        ]

        # Make sure we don't have duplicates
        known_companies = list(set(known_companies))

        # Return the requested number of companies
        return known_companies[:num_startups]

    def extract_company_name(self, soup, url):
        """Extract company name using multiple methods."""
        # Method 1: Try multiple CSS selectors
        for selector in ["h1", ".company-header h1", ".company-name", "title", ".name", "h2"]:
            name_element = soup.select_one(selector)
            if name_element:
                name = name_element.text.strip()
                # Clean up common patterns in titles
                name = name.replace(" - Y Combinator", "").replace(" | YC", "").replace(" | Y Combinator", "").strip()
                if name and name != "Y Combinator" and len(name) < 50:
                    return name

        # Method 2: Extract from URL
        company_slug = url.split("/")[-1]
        if company_slug:
            # Convert slug to title case (e.g., "company-name" to "Company Name")
            name = " ".join(word.capitalize() for word in company_slug.split("-"))
            return name

        # If all else fails
        return "Unknown Company"

    def extract_description(self, soup):
        """Extract company description using multiple methods."""
        # Method 1: Try multiple CSS selectors
        for selector in [".prose p", ".company-description", "section p", "meta[name='description']",
                         ".about", "#about", "[itemprop='description']"]:
            if selector.startswith("meta"):
                meta = soup.select_one(selector)
                if meta and meta.get("content"):
                    return meta.get("content").strip()
            else:
                desc_elements = soup.select(selector)
                if desc_elements:
                    # Join multiple paragraphs if found
                    combined_desc = " ".join([p.text.strip() for p in desc_elements[:3]])
                    if len(combined_desc) > 20:  # Ensure it's substantive
                        return combined_desc

        # Method 2: Look for paragraphs near the company name/header
        h1_element = soup.find("h1")
        if h1_element:
            # Look at the next few elements after the h1
            current = h1_element.next_sibling
            paragraphs = []
            count = 0
            while current and count < 5:
                if hasattr(current, 'name') and current.name == 'p':
                    paragraphs.append(current.text.strip())
                count += 1
                current = current.next_sibling

            if paragraphs:
                return " ".join(paragraphs)

        # Method 3: Just grab all paragraphs and use the longest one
        all_paragraphs = soup.find_all("p")
        if all_paragraphs:
            paragraphs_text = [p.text.strip() for p in all_paragraphs if len(p.text.strip()) > 50]
            if paragraphs_text:
                return max(paragraphs_text, key=len)

        # If all else fails
        return "No description available"

    def extract_tags(self, soup, description):
        """Extract or infer tags for the company."""
        # Method 1: Try multiple CSS selectors
        tags = []
        tag_selectors = [
            ".TagList_tag__yhj5n", ".tag", ".label", "[data-component='Tag']",
            ".category", ".industry", ".sector"
        ]

        for selector in tag_selectors:
            tag_elements = soup.select(selector)
            if tag_elements:
                tags = [tag.text.strip() for tag in tag_elements]
                break

        # Method 2: Look for keywords in the description
        if not tags:
            # Common YC startup categories
            potential_tags = [
                "AI", "Machine Learning", "SaaS", "Fintech", "Financial", "Healthcare",
                "Education", "B2B", "Enterprise", "Consumer", "Marketplace", "Mobile",
                "Hardware", "Climate", "Bio", "Developer Tools", "Crypto", "Security",
                "Blockchain", "Web3", "E-commerce", "Robotics", "IoT", "Analytics"
            ]
            description_lower = description.lower()
            for tag in potential_tags:
                if tag.lower() in description_lower:
                    tags.append(tag)

        # Always return a string
        return ", ".join(tags) if tags else "Unspecified"

    def extract_founding_year(self, soup):
        """Extract founding year information."""
        # Method 1: Try to find explicit founding year text
        year_patterns = [
            r"[Ff]ounded in (\d{4})",
            r"[Ee]st\. (\d{4})",
            r"[Ee]stablished (\d{4})",
            r"[Ss]ince (\d{4})",
            r"[Ff]ounded: (\d{4})"
        ]

        for pattern in year_patterns:
            match = re.search(pattern, soup.text)
            if match:
                return match.group(1)

        # Method 2: Look for years in specific elements
        year_elements = soup.select(".founded, .year, .founding-date")
        if year_elements:
            for el in year_elements:
                years = re.findall(r"\b(20\d\d|19\d\d)\b", el.text)
                if years:
                    return years[0]

        # Method 3: Look for any year between 2005-current year
        current_year = datetime.datetime.now().year
        # Look for years in the YC range (2005 onward)
        all_years = re.findall(r"\b(20\d\d|19\d\d)\b", soup.text)
        plausible_years = [y for y in all_years if 2005 <= int(y) <= current_year]
        if plausible_years:
            # Assume the earliest year is the founding year
            return min(plausible_years)

        return "Unknown"

    def extract_startup_info(self, url: str) -> Dict[str, Any]:
        """Extract information from a startup's page."""
        soup = self.fetch_page(url)
        if not soup:
            return self.extract_info_from_url(url)  # Fallback to extracting from URL

        company_name = self.extract_company_name(soup, url)
        description = self.extract_description(soup)
        tags = self.extract_tags(soup, description)
        founding_year = self.extract_founding_year(soup)

        # Print extraction results for debugging
        print(f"\nExtraction results for {url}:")
        print(f"- Company name: {company_name}")
        print(f"- Description length: {len(description)} characters")
        print(f"- Tags: {tags}")
        print(f"- Founding year: {founding_year}")

        return {
            "company_name": company_name,
            "description": description,
            "tags": tags,
            "founding_year": founding_year,
            "url": url
        }

    def extract_info_from_url(self, url):
        """Fallback method to extract minimal information from URL."""
        # Get company name from URL
        company_slug = url.split("/")[-1]
        if company_slug:
            # Convert slug to title case (e.g., "company-name" to "Company Name")
            name = " ".join(word.capitalize() for word in company_slug.split("-"))
        else:
            name = "Unknown Company"

        return {
            "company_name": name,
            "description": "Information could not be extracted from the company page.",
            "tags": "Unspecified",
            "founding_year": "Unknown",
            "url": url
        }

    def generate_project_idea(self, company_info: Dict[str, Any]) -> str:
        """Generate a project idea using Claude API or fallback to a rule-based approach."""
        print(f"Generating project idea for {company_info['company_name']}...")

        # Try using Claude API if available
        if self.claude_client is not None:
            try:
                prompt = f"""
                Company: {company_info['company_name']}
                Description: {company_info['description']}
                Tags: {company_info['tags']}

                Suggest in one sentence a project idea related to this company's domain:
                """

                response = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20240229",
                    max_tokens=100,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                # Extract the response content
                if hasattr(response, 'content') and response.content:
                    idea = response.content[0].text.strip()
                    # Remove line breaks and ensure it's a single sentence
                    idea = idea.replace('\n', ' ').strip()
                    if idea:
                        print(f"Generated idea using Claude API: {idea}")
                        return idea

            except Exception as e:
                print(f"Error using Claude API: {e}")
                print("Falling back to rule-based idea generation...")

        # Fallback to rule-based generation
        company_name = company_info['company_name']
        tags = company_info['tags'].lower()
        description = company_info['description'].lower()

        # Generate ideas based on tags and description keywords
        ideas = []

        if any(tech in tags or tech in description for tech in
               ["ai", "machine learning", "ml", "artificial intelligence"]):
            ideas.extend([
                f"Create a custom ML model to enhance {company_name}'s data processing capabilities.",
                f"Build a recommendation engine for {company_name}'s product using collaborative filtering."
            ])

        if any(tech in tags or tech in description for tech in ["saas", "software", "platform"]):
            ideas.extend([
                f"Develop an integration platform connecting {company_name} with other popular SaaS tools.",
                f"Create a browser extension that enhances the {company_name} user experience."
            ])

        if any(tech in tags or tech in description for tech in ["fintech", "finance", "banking", "payment"]):
            ideas.extend([
                f"Build a financial data visualization dashboard that works with {company_name}'s API.",
                f"Create a budget planning tool that integrates with {company_name}'s platform."
            ])

        if any(tech in tags or tech in description for tech in ["health", "medical", "healthcare"]):
            ideas.extend([
                f"Develop a patient-facing mobile app that connects to {company_name}'s healthcare platform.",
                f"Create a health data analytics tool that complements {company_name}'s service."
            ])

        # Add generic ideas if none match or as additional options
        generic_ideas = [
            f"Build a marketplace connecting {company_name} users with service providers.",
            f"Create a data analytics dashboard for {company_name}'s industry.",
            f"Develop a mobile app extending {company_name}'s functionality.",
            f"Design a plugin that integrates {company_name} with popular work tools.",
            f"Build an API wrapper for {company_name}'s service to extend its capabilities."
        ]

        # Combine specific and generic ideas
        all_ideas = ideas + generic_ideas

        # Return a random idea if we have any, otherwise return a default message
        if all_ideas:
            chosen_idea = random.choice(all_ideas)
            print(f"Generated idea using rule-based approach: {chosen_idea}")
            return chosen_idea
        else:
            return f"Create a companion analytics tool for {company_name}'s service."

    def scrape_startups(self, num_startups: int = 10) -> List[Dict[str, Any]]:
        """Scrape information for multiple startups."""
        startup_links = self.get_startup_links(num_startups)
        all_startup_info = []

        for i, link in enumerate(startup_links):
            try:
                print(f"Scraping {i + 1}/{len(startup_links)}: {link}")
                startup_info = self.extract_startup_info(link)

                if startup_info:
                    # Generate project idea
                    project_idea = self.generate_project_idea(startup_info)
                    startup_info["project_idea"] = project_idea
                    all_startup_info.append(startup_info)

                # Be polite to the server with a delay between requests
                time.sleep(random.uniform(1, 3))
            except Exception as e:
                print(f"Error processing {link}: {e}")
                # Continue with the next link

        return all_startup_info

    def save_to_csv(self, startup_data: List[Dict[str, Any]]) -> None:
        """Save startup data to a CSV file."""
        if not startup_data:
            print("No data to save.")
            return

        fieldnames = ["company_name", "description", "tags", "founding_year", "project_idea", "url"]

        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for startup in startup_data:
                    # Filter the dictionary to only include our fieldnames
                    filtered_data = {k: startup.get(k, '') for k in fieldnames}
                    writer.writerow(filtered_data)

            print(f"Data saved to {self.output_file}")

            # Also save as JSON for easier programmatic access
            json_file = self.output_file.replace('.csv', '.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(startup_data, f, indent=2)

            print(f"Data also saved as JSON to {json_file}")

        except Exception as e:
            print(f"Error saving data: {e}")

            # Try to save to a backup file
            backup_file = f"backup_{int(time.time())}_{self.output_file}"
            try:
                with open(backup_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for startup in startup_data:
                        filtered_data = {k: startup.get(k, '') for k in fieldnames}
                        writer.writerow(filtered_data)
                print(f"Data saved to backup file: {backup_file}")
            except Exception as backup_error:
                print(f"Could not save backup file either: {backup_error}")
                # As a last resort, print the data
                print("\nData collected (not saved to file):")
                for startup in startup_data:
                    print(json.dumps(startup, indent=2))


def load_api_key_from_file(file_path="api.txt"):
    """Load API key from a text file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                # Read the first line and strip any whitespace
                api_key = file.readline().strip()
                if api_key:
                    print(f"Successfully loaded API key from {file_path}")
                    return api_key
                else:
                    print(f"API key file {file_path} is empty")
        else:
            print(f"API key file {file_path} not found")
    except Exception as e:
        print(f"Error loading API key from file: {e}")

    return None


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Scrape Y Combinator startup data')
    parser.add_argument('-n', '--num', type=int, default=15, help='Number of startups to scrape')
    parser.add_argument('-o', '--output', type=str, default='yc_startups.csv', help='Output CSV filename')
    parser.add_argument('--claude-api-key', type=str, help='Claude (Anthropic) API key for project suggestions')
    parser.add_argument('--api-file', type=str, default='api.txt', help='File containing Claude API key')
    args = parser.parse_args()

    # First try to get API key from command line
    api_key = args.claude_api_key

    # If not provided via command line, try to load from file
    if not api_key:
        api_key = load_api_key_from_file(args.api_file)
        if not api_key:
            print("No API key provided. Will use rule-based approach for project ideas.")

    # Create scraper instance with API key
    scraper = YCScraper(output_file=args.output, claude_api_key=api_key)

    # Scrape startup data
    print(f"Starting to scrape {args.num} YC startups...")
    startup_data = scraper.scrape_startups(args.num)

    # Save data to CSV
    scraper.save_to_csv(startup_data)

    # Print preview
    print("\nPreview of collected data:")
    for i, startup in enumerate(startup_data[:3]):
        print(f"\n--- Startup {i + 1} ---")
        print(f"Name: {startup['company_name']}")
        print(f"Description: {startup['description'][:100]}...")
        print(f"Tags: {startup['tags']}")
        print(f"Project Idea: {startup['project_idea']}")

    print(f"\nTotal startups scraped: {len(startup_data)}")
    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()