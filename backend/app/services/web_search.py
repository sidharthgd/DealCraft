import logging
import json
import requests
import aiohttp
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings

logger = logging.getLogger(__name__)

class WebSearchService:
    """Service for performing web searches to gather biographical information."""
    
    def __init__(self):
        self.google_api_key = settings.GOOGLE_SEARCH_API_KEY
        self.google_search_engine_id = settings.GOOGLE_SEARCH_ENGINE_ID
        self.google_search_url = "https://www.googleapis.com/customsearch/v1"
        
        if self.google_api_key and self.google_search_engine_id:
            logger.info("WebSearchService initialized with Google Custom Search API")
            self.search_enabled = True
        else:
            logger.warning("Google Custom Search API not configured - web search will return empty results")
            logger.info("To enable web search, set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables")
            self.search_enabled = False
    
    async def search_executive_bio(
        self, 
        executive_name: str, 
        company_name: Optional[str] = None,
        company_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search for biographical information about an executive."""
        try:
            # Construct search queries for different types of information
            search_queries = self._build_search_queries(executive_name, company_name, company_location)
            
            bio_data = {
                "name": executive_name,
                "education": [],
                "experience": [],
                "raw_results": []
            }
            
            # Perform searches for each query
            for query_type, query in search_queries.items():
                logger.info(f"Searching for {query_type}: {query}")
                results = await self._perform_search(query)
                
                # Process results based on query type
                if query_type == "education":
                    education_info = self._extract_education_info(results, executive_name)
                    bio_data["education"].extend(education_info)
                elif query_type == "experience":
                    experience_info = self._extract_experience_info(results, executive_name)
                    bio_data["experience"].extend(experience_info)
                
                bio_data["raw_results"].append({
                    "query_type": query_type,
                    "query": query,
                    "results": results[:3]  # Keep top 3 results for debugging
                })
            
            # De-duplicate and lightly normalize
            bio_data["education"] = self._dedupe_education(bio_data["education"])
            bio_data["experience"] = self._dedupe_experience(bio_data["experience"])
            
            return bio_data
            
        except Exception as e:
            logger.error(f"Error searching for executive bio {executive_name}: {e}")
            return {
                "name": executive_name,
                "education": [],
                "experience": [],
                "error": str(e)
            }
    
    def _build_search_queries(
        self, 
        executive_name: str, 
        company_name: Optional[str] = None,
        company_location: Optional[str] = None
    ) -> Dict[str, str]:
        """Build targeted search queries for different types of biographical information."""
        queries: Dict[str, str] = {}
        
        # Education-focused queries
        queries["education"] = f'"{executive_name}" education university college degree MBA'
        if company_name:
            queries["education"] += f' "{company_name}"'
        
        # Experience-focused queries
        queries["experience"] = f'"{executive_name}" career experience jobs companies'
        if company_name:
            queries["experience"] += f' "{company_name}"'
        
        # LinkedIn-specific search (via search results, not direct scraping)
        queries["linkedin"] = f'"{executive_name}" site:linkedin.com'
        if company_name:
            queries["linkedin"] += f' "{company_name}"'
        
        # General biographical search
        queries["biography"] = f'"{executive_name}" biography background'
        if company_name:
            queries["biography"] += f' "{company_name}" CEO founder'
        
        return queries
    
    async def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a web search using Google Custom Search API."""
        if not self.search_enabled:
            logger.info(f"Search disabled - skipping query: {query}")
            return []
        
        try:
            params = {
                'key': self.google_api_key,
                'cx': self.google_search_engine_id,
                'q': query,
                'num': 8,  # Increase to improve recall (max 10)
                'safe': 'active',
                'hl': 'en',
                'fields': 'items(title,snippet,link,displayLink)'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.google_search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        
                        results: List[Dict[str, Any]] = []
                        for item in items:
                            results.append({
                                'title': item.get('title', ''),
                                'snippet': item.get('snippet', ''),
                                'url': item.get('link', ''),
                                'source': item.get('displayLink', '')
                            })
                        
                        logger.info(f"Google search returned {len(results)} results for query: {query}")
                        return results
                    
                    if response.status == 429:
                        logger.warning(f"Google Search API rate limit exceeded for query: {query}")
                        return []
                    
                    logger.error(f"Google Search API error {response.status} for query: {query}")
                    return []
            
        except aiohttp.ClientError as e:
            logger.error(f"Network error during Google search for query '{query}': {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Google search for query '{query}': {e}")
            return []
    
    def _extract_education_info(self, search_results: List[Dict[str, Any]], executive_name: str) -> List[Dict[str, str]]:
        """Extract education information from search results."""
        education_info: List[Dict[str, str]] = []
        
        for result in search_results:
            orig_snippet = result.get("snippet", "")
            orig_title = result.get("title", "")
            sniff = orig_snippet.lower()
            tt = orig_title.lower()
            
            education_keywords = [
                "university", "college", "bachelor", "master", "mba", "phd",
                "degree", "graduated", "school", "institute", "academy"
            ]
            
            if any(keyword in sniff or keyword in tt for keyword in education_keywords):
                education_details = self._parse_education_from_text(orig_snippet + " " + orig_title)
                education_info.extend(education_details)
        
        return education_info
    
    def _extract_experience_info(self, search_results: List[Dict[str, Any]], executive_name: str) -> List[Dict[str, str]]:
        """Extract work experience information from search results."""
        experience_info: List[Dict[str, str]] = []
        
        for result in search_results:
            orig_snippet = result.get("snippet", "")
            orig_title = result.get("title", "")
            sniff = orig_snippet.lower()
            tt = orig_title.lower()
            
            experience_keywords = [
                "worked at", "ceo", "founder", "president", "vice president",
                "director", "manager", "company", "corporation", "before", "joined", "served as"
            ]
            
            if any(keyword in sniff or keyword in tt for keyword in experience_keywords):
                experience_details = self._parse_experience_from_text(orig_snippet + " " + orig_title)
                experience_info.extend(experience_details)
        
        return experience_info
    
    def _parse_education_from_text(self, text: str) -> List[Dict[str, str]]:
        """Parse education information from text snippets using generic pattern matching."""
        education: List[Dict[str, str]] = []
        
        try:
            import re
            
            # Enhanced patterns for education parsing
            education_patterns = [
                r'graduated\s+from\s+([^,.\n]+?(?:university|college|institute|school))[^,.\n]*?with\s+(?:a\s+)?([^,.\n]*(?:bachelor|master|mba|phd|degree)[^,.\n]*)',
                r'received\s+(?:a\s+|his\s+|her\s+)?([^,.\n]*(?:bachelor|master|mba|phd|degree)[^,.\n]*)\s+from\s+([^,.\n]+?(?:university|college|institute|school))',
                r'holds\s+(?:a\s+)?([^,.\n]*(?:bachelor|master|mba|phd|degree)[^,.\n]*)\s+from\s+([^,.\n]+?(?:university|college|institute|school))',
                r'earned\s+(?:a\s+|his\s+|her\s+)?([^,.\n]*(?:bachelor|master|mba|phd|degree)[^,.\n]*)\s+from\s+([^,.\n]+?(?:university|college|institute|school))',
                r'([^,.\n]+?(?:university|college|institute|school))[^,.\n]*?[-–]\s*([^,.\n]*(?:bachelor|master|mba|phd|degree)[^,.\n]*)',
                r'studied\s+at\s+([^,.\n]+?(?:university|college|institute|school))[^,.\n]*?(?:where\s+he\s+|where\s+she\s+)?(?:received\s+|earned\s+)?(?:a\s+)?([^,.\n]*(?:bachelor|master|mba|phd|degree)[^,.\n]*)',
            ]
            
            def attach_major(base_degree: str, context: str) -> str:
                # Find " in X" or " in the field of X"
                m = re.search(r'\b(?:in|with\s+a\s+concentration\s+in)\s+([A-Za-z][A-Za-z &/\-]+)', context, flags=re.IGNORECASE)
                if m:
                    major = m.group(1).strip().rstrip('.,')
                    # Avoid duplicating if base_degree already contains the major phrase
                    if major.lower() not in base_degree.lower():
                        return f"{base_degree} in {major}"
                return base_degree
            
            for pattern in education_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        group1, group2 = match.group(1).strip(), match.group(2).strip()
                        if any(k in group1.lower() for k in ['university', 'college', 'institute', 'school']):
                            school, degree = group1, group2
                        else:
                            school, degree = group2, group1
                        
                        degree_norm = degree
                        # Normalize common degree phrasing
                        if 'bachelor' in degree.lower():
                            base = "Bachelor's"
                            degree_norm = attach_major(base, degree)
                        elif 'master' in degree.lower() and 'mba' not in degree.lower():
                            base = "Master's"
                            degree_norm = attach_major(base, degree)
                        elif 'mba' in degree.lower():
                            degree_norm = 'MBA'
                        elif 'phd' in degree.lower():
                            degree_norm = 'PhD'
                        
                        education.append({
                            "type": "graduate" if any(x in degree_norm.lower() for x in ['master', 'mba', 'phd']) else "undergraduate",
                            "school": school.strip().rstrip(' .,'),
                            "degree": degree_norm.strip().rstrip(' .,'),
                            "year": ""
                        })
        
        except Exception as e:
            logger.error(f"Error parsing education from text: {e}")
        
        return education
    
    def _parse_experience_from_text(self, text: str) -> List[Dict[str, str]]:
        """Parse work experience from text snippets using generic pattern matching."""
        experience: List[Dict[str, str]] = []
        
        try:
            import re
            
            experience_patterns = [
                r'(?:worked\s+at|employed\s+at|joined)\s+([^,.\n]+?)(?:\s+as\s+|\s+in\s+|\s+,\s*)([^,.\n]+?)(?:\s+from\s+|\s+\()?(\d{4}[-–]\d{4}|\d{4}[-–]present|since\s+\d{4})',
                r'(ceo|president|founder|co-founder|chief\s+executive|director|manager|vice\s+president|vp)\s+(?:at|of|for)\s+([^,.\n]+?)(?:\s+from\s+|\s+\()?(\d{4}[-–]\d{4}|\d{4}[-–]present|since\s+\d{4})',
                r'served\s+as\s+([^,.\n]+?)\s+(?:at|for)\s+([^,.\n]+?)(?:\s+from\s+|\s+\()?(\d{4}[-–]\d{4}|\d{4}[-–]present|since\s+\d{4})',
                r'([^,.\n]+?(?:inc\.|corp\.|llc|corporation|company))[^,.\n]*?[-–]\s*([^,.\n]+?)(?:\s+\()?(\d{4}[-–]\d{4}|\d{4}[-–]present|since\s+\d{4})',
                # Patterns without explicit years
                r'(?:is\s+the\s+|was\s+the\s+|became\s+)?(?:current\s+)?(ceo|president|founder|co-founder|chief\s+executive|director)\s+(?:at|of|for)\s+([^,.\n]+)'
            ]
            
            for pattern in experience_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        # Determine placement of company/title/years
                        if any(role in groups[0].lower() for role in ['ceo', 'president', 'founder', 'co-founder', 'chief', 'director', 'manager', 'vice']):
                            title = groups[0]
                            company = groups[1]
                            years = groups[2] if len(groups) > 2 else ""
                        else:
                            company = groups[0]
                            title = groups[1]
                            years = groups[2] if len(groups) > 2 else ""
                        
                        # Normalize years like "since 2011" -> "2011-present"
                        if years and years.lower().startswith('since'):
                            yr = re.sub(r'[^0-9]', '', years)
                            if len(yr) == 4:
                                years = f"{yr}-present"
                        
                        company = company.strip().rstrip(' .,' )
                        title = title.strip().rstrip(' .,' )
                        years = years.strip()
                        
                        if len(company) > 2 and len(title) > 1:
                            experience.append({
                                "company": company,
                                "title": title,
                                "years": years
                            })
        
        except Exception as e:
            logger.error(f"Error parsing experience from text: {e}")
        
        return experience
    
    def _dedupe_education(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen: set[Tuple[str, str]] = set()
        deduped: List[Dict[str, str]] = []
        for it in items:
            key = (it.get('school', '').strip().lower(), it.get('degree', '').strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)
        return deduped
    
    def _dedupe_experience(self, items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen: set[Tuple[str, str, str]] = set()
        deduped: List[Dict[str, str]] = []
        for it in items:
            key = (
                it.get('company', '').strip().lower(),
                it.get('title', '').strip().lower(),
                it.get('years', '').strip().lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)
        return deduped
    
    def format_bio_section(self, bio_data: Dict[str, Any]) -> str:
        """Format the biographical data into the required SFE Bio format."""
        if not bio_data or bio_data.get("error"):
            return "Biographical information not available."
        
        name = bio_data.get("name", "Executive")
        education = bio_data.get("education", [])
        experience = bio_data.get("experience", [])
        
        formatted_bio = f"**{name}**\n\n"
        
        # Format education section
        formatted_bio += "**School(s) attended:**\n"
        
        undergraduate = [edu for edu in education if edu.get("type") == "undergraduate"]
        graduate = [edu for edu in education if edu.get("type") == "graduate"]
        
        if undergraduate:
            for ug in undergraduate:
                degree = ug.get("degree", "")
                school = ug.get("school", "")
                year = ug.get("year", "")
                formatted_bio += f"• {school} - {degree}"
                if year:
                    formatted_bio += f" ({year})"
                formatted_bio += "\n"
        else:
            formatted_bio += "• Undergraduate education: Not available in search results\n"
        
        if graduate:
            for grad in graduate:
                degree = grad.get("degree", "")
                school = grad.get("school", "")
                year = grad.get("year", "")
                formatted_bio += f"• {school} - {degree}"
                if year:
                    formatted_bio += f" ({year})"
                formatted_bio += "\n"
        
        # Format experience section
        formatted_bio += "\n**Full-time jobs since UG graduation:**\n"
        
        if experience:
            sorted_experience = sorted(experience, key=lambda x: x.get("years", ""))
            for exp in sorted_experience:
                company = exp.get("company", "")
                title = exp.get("title", "")
                years = exp.get("years", "")
                formatted_bio += f"• {company}"
                if title:
                    formatted_bio += f" - {title}"
                if years:
                    formatted_bio += f" ({years})"
                formatted_bio += "\n"
        else:
            formatted_bio += "• Professional experience: Not available in search results\n"
        
        return formatted_bio


# Global instance
web_search_service = WebSearchService()
