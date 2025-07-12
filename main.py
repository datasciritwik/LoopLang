import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import re
import random
from crawl4ai import AsyncWebCrawler
from ddgs import DDGS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Goal status enum
class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

# State structure for the agent
@dataclass
class AgentState:
    goal: str
    target_count: int
    current_results: List[Dict[str, Any]]
    search_queries: List[str]
    attempted_sources: List[str]
    status: GoalStatus
    iteration_count: int
    max_iterations: int
    messages: List[Any]
    last_error: Optional[str] = None

class PersistentSearchAgent:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            model="llama3-70b-8192",  # Fixed model name
            api_key=groq_api_key,
            temperature=0.3
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.ddgs = DDGS()
        
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform web search using DuckDuckGo"""
        try:
            results = list(self.ddgs.text(query, safesearch='off', max_results=max_results))
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    async def crawler(self, url: str) -> str:
        """Crawl and extract content from a URL"""
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
                return result.markdown
        except Exception as e:
            print(f"Crawling error for {url}: {e}")
            return ""
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes - all nodes need to be async-compatible
        workflow.add_node("analyze_goal", self._analyze_goal)
        workflow.add_node("generate_search_strategy", self._generate_search_strategy)
        workflow.add_node("execute_search", self._execute_search)
        workflow.add_node("validate_results", self._validate_results)
        workflow.add_node("refine_strategy", self._refine_strategy)
        workflow.add_node("finalize_results", self._finalize_results)
        
        # Add edges
        workflow.add_edge("analyze_goal", "generate_search_strategy")
        workflow.add_edge("generate_search_strategy", "execute_search")
        workflow.add_edge("execute_search", "validate_results")
        
        # Conditional edges based on validation
        workflow.add_conditional_edges(
            "validate_results",
            self._should_continue,
            {
                "continue": "refine_strategy",
                "complete": "finalize_results", 
                "failed": END
            }
        )
        
        workflow.add_edge("refine_strategy", "execute_search")
        workflow.add_edge("finalize_results", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_goal")
        
        return workflow.compile(checkpointer=self.memory)
    
    def _analyze_goal(self, state: AgentState) -> AgentState:
        """Analyze the goal and extract requirements"""
        print(f"🎯 Analyzing goal: {state.goal}")
        
        analysis_prompt = ChatPromptTemplate.from_template("""
        Analyze this goal and extract key information:
        Goal: {goal}
        
        Please identify:
        1. What type of content is needed (websites, emails, job links, etc.)
        2. How many items are required
        3. What specific criteria should be met
        4. What search strategies would be most effective
        
        Respond in JSON format with keys: content_type, quantity, criteria, search_strategies
        """)
        
        response = self.llm.invoke(analysis_prompt.format(goal=state.goal))
        
        try:
            analysis = json.loads(response.content)
            state.target_count = analysis.get("quantity", 10)
            state.messages.append(AIMessage(content=f"Goal analyzed: Need {state.target_count} {analysis.get('content_type', 'items')}"))
        except:
            state.target_count = 10  # Default fallback
            
        state.status = GoalStatus.IN_PROGRESS
        return state
    
    def _generate_search_strategy(self, state: AgentState) -> AgentState:
        """Generate search queries and strategies"""
        print(f"🔍 Generating search strategy for: {state.goal}")
        
        strategy_prompt = ChatPromptTemplate.from_template("""
        Generate 5-10 diverse search queries for this goal: {goal}
        
        Current results count: {current_count}/{target_count}
        Previously tried queries: {previous_queries}
        
        Create search queries that:
        1. Are specific and targeted
        2. Use different keywords and approaches
        3. Haven't been tried before
        4. Are likely to yield the required content type
        
        Return as a JSON list of strings.
        """)
        
        response = self.llm.invoke(strategy_prompt.format(
            goal=state.goal,
            current_count=len(state.current_results),
            target_count=state.target_count,
            previous_queries=state.search_queries
        ))
        
        try:
            new_queries = json.loads(response.content)
            # Add new queries that haven't been tried
            for query in new_queries:
                if query not in state.search_queries:
                    state.search_queries.append(query)
        except:
            # Fallback queries based on goal type
            if "website" in state.goal.lower():
                state.search_queries.extend([
                    f"best {state.goal.split()[0]} websites",
                    f"top {state.goal.split()[0]} tools online",
                    f"useful {state.goal.split()[0]} resources"
                ])
            elif "email" in state.goal.lower():
                state.search_queries.extend([
                    "recruiter email contacts",
                    "HR manager email directory",
                    "talent acquisition email list"
                ])
        
        print(f"📝 Generated {len(state.search_queries)} search queries")
        return state
    
    async def _execute_search(self, state: AgentState) -> AgentState:
        """Execute search queries and collect results"""
        print(f"🔎 Executing search (Iteration {state.iteration_count + 1})")
        
        # Take next few unused queries
        queries_to_try = [q for q in state.search_queries if q not in state.attempted_sources][:3]
        
        for query in queries_to_try:
            try:
                print(f"  Searching: {query}")
                results = await self._perform_search(query, state.goal)
                
                # Add unique results
                for result in results:
                    # Check for uniqueness based on URL or email
                    url = result.get('url', '')
                    email = result.get('email', '')
                    
                    is_unique = True
                    for existing in state.current_results:
                        if url and url == existing.get('url'):
                            is_unique = False
                            break
                        if email and email == existing.get('email'):
                            is_unique = False
                            break
                    
                    if is_unique:
                        state.current_results.append(result)
                
                state.attempted_sources.append(query)
                
            except Exception as e:
                print(f"  Error searching '{query}': {str(e)}")
                state.last_error = str(e)
                state.attempted_sources.append(query)
        
        state.iteration_count += 1
        print(f"📊 Current results: {len(state.current_results)}/{state.target_count}")
        return state
    
    def _extract_emails_from_content(self, content: str) -> List[str]:
        """Extract email addresses from content"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        return list(set(emails))  # Remove duplicates
    
    def _extract_job_info_from_content(self, content: str, url: str) -> Dict[str, Any]:
        """Extract job information from content"""
        job_info = {}
        
        # Extract job title (simple heuristic)
        title_patterns = [
            r'(?i)<title>(.+?)\s*[-|]\s*(.+?)</title>',
            r'(?i)job title:\s*(.+?)(?:\n|$)',
            r'(?i)position:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content)
            if match:
                job_info['job_title'] = match.group(1).strip()
                break
        
        # Extract company name
        company_patterns = [
            r'(?i)company:\s*(.+?)(?:\n|$)',
            r'(?i)employer:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, content)
            if match:
                job_info['company'] = match.group(1).strip()
                break
        
        # Extract location
        location_patterns = [
            r'(?i)location:\s*(.+?)(?:\n|$)',
            r'(?i)based in:\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, content)
            if match:
                job_info['location'] = match.group(1).strip()
                break
        
        job_info['job_url'] = url
        return job_info
    
    def _get_search_tricks(self, goal: str) -> List[str]:
        """Get search tricks based on goal type"""
        tricks = []
        
        if "email" in goal.lower():
            tricks.extend([
                "site:linkedin.com",
                "\"@company.com\"",
                "contact email",
                "recruiter email",
                "hiring manager"
            ])
        
        if "website" in goal.lower():
            tricks.extend([
                "best tools 2024",
                "top resources",
                "list of sites",
                "directory"
            ])
        
        if "job" in goal.lower():
            tricks.extend([
                "site:indeed.com",
                "site:glassdoor.com",
                "site:linkedin.com/jobs",
                "hiring",
                "careers"
            ])
        
        return tricks
    
    def _is_relevant_result(self, result: Dict[str, Any], goal: str) -> bool:
        """Check if a search result is relevant to the goal"""
        title = result.get('title', '').lower()
        body = result.get('body', '').lower()
        url = result.get('href', '').lower()
        
        goal_lower = goal.lower()
        
        # Check for relevant keywords
        if "email" in goal_lower:
            return any(keyword in title or keyword in body for keyword in ['email', 'contact', 'recruiter', 'hiring'])
        
        if "website" in goal_lower:
            return any(keyword in title or keyword in body for keyword in ['tool', 'platform', 'service', 'website'])
        
        if "job" in goal_lower:
            return any(keyword in title or keyword in body for keyword in ['job', 'career', 'position', 'hiring'])
        
        return True  # Default to relevant if unsure
    
    async def _perform_search(self, query: str, goal: str) -> List[Dict[str, Any]]:
        """Perform actual search and extract results with web crawling"""
        results = []
        
        try:
            # Get search tricks for specific purposes
            search_tricks = self._get_search_tricks(goal)
            
            # Combine base query with search tricks
            enhanced_queries = [query]
            
            # Add enhanced queries with search tricks
            for trick in search_tricks[:3]:  # Use top 3 tricks
                enhanced_queries.append(f"{query} {trick}")
            
            # Perform searches
            for enhanced_query in enhanced_queries[:2]:  # Limit to 2 queries per iteration
                print(f"  Searching: {enhanced_query}")
                search_results = self.search(enhanced_query, max_results=5)
                
                for result in search_results:
                    if self._is_relevant_result(result, goal):
                        # Basic result info
                        processed_result = {
                            'title': result.get('title', ''),
                            'url': result.get('href', ''),
                            'description': result.get('body', ''),
                            'source': 'web_search'
                        }
                        
                        # Crawl the page for additional content
                        try:
                            url = processed_result['url']
                            if url:
                                print(f"    Crawling: {url}")
                                content = await self.crawler(url)
                                
                                if content:
                                    # Extract specific information based on goal
                                    if any(keyword in goal.lower() for keyword in ['email', 'contact', 'recruiter']):
                                        emails = self._extract_emails_from_content(content)
                                        if emails:
                                            processed_result['emails'] = emails
                                            processed_result['contact_info'] = emails
                                    
                                    elif any(keyword in goal.lower() for keyword in ['job', 'career', 'position', 'hiring']):
                                        job_info = self._extract_job_info_from_content(content, url)
                                        processed_result.update(job_info)
                                    
                                    else:
                                        # For general content, extract key information
                                        processed_result['content'] = content[:500]  # First 500 chars
                                    
                                    results.append(processed_result)
                                    
                        except Exception as e:
                            print(f"    Error crawling {processed_result['url']}: {e}")
                            # Still add the basic result even if crawling fails
                            results.append(processed_result)
                        
                        # Add delay between crawls
                        await asyncio.sleep(random.uniform(1, 2))
                
                # Add delay between searches
                await asyncio.sleep(random.uniform(2, 4))
                
        except Exception as e:
            print(f"Search error: {e}")
            # Fallback to mock data for demonstration
            results.extend(self._get_fallback_results(goal))
        
        return results
    
    def _get_fallback_results(self, goal: str) -> List[Dict[str, Any]]:
        """Fallback mock results when search fails"""
        if "website" in goal.lower() and "digital marketing" in goal.lower():
            return [
                {"title": "HubSpot Marketing Hub", "url": "https://hubspot.com", "description": "All-in-one marketing platform"},
                {"title": "Moz SEO Tools", "url": "https://moz.com", "description": "SEO and marketing analytics"},
                {"title": "SEMrush", "url": "https://semrush.com", "description": "Digital marketing toolkit"},
            ]
        elif "email" in goal.lower() and "recruiter" in goal.lower():
            return [
                {"name": "Sarah Johnson", "email": "sarah.johnson@techcorp.com", "company": "TechCorp", "role": "Senior Recruiter"},
                {"name": "Mike Davis", "email": "mike.davis@startup.io", "company": "StartupIO", "role": "Talent Acquisition Manager"},
            ]
        elif "job" in goal.lower():
            return [
                {"title": "Senior Software Engineer", "company": "TechCorp", "url": "https://jobs.techcorp.com/123", "location": "Remote"},
                {"title": "Data Scientist", "company": "DataCorp", "url": "https://careers.datacorp.com/456", "location": "New York"},
            ]
        return []
    
    def _validate_results(self, state: AgentState) -> AgentState:
        """Validate if results meet quality and quantity requirements"""
        print(f"✅ Validating results: {len(state.current_results)}/{state.target_count}")
        
        # Check if we have enough results
        if len(state.current_results) >= state.target_count:
            state.status = GoalStatus.COMPLETED
            print("🎉 Goal achieved!")
            return state
        
        # Check if we've hit max iterations
        if state.iteration_count >= state.max_iterations:
            state.status = GoalStatus.FAILED
            print("❌ Max iterations reached without achieving goal")
            return state
        
        # Check if we have more search strategies to try
        untried_queries = [q for q in state.search_queries if q not in state.attempted_sources]
        if len(untried_queries) == 0 and len(state.current_results) < state.target_count:
            # Need to generate more strategies
            print("🔄 Need to generate more search strategies")
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine if the agent should continue, complete, or fail"""
        if state.status == GoalStatus.COMPLETED:
            return "complete"
        elif state.status == GoalStatus.FAILED:
            return "failed"
        else:
            return "continue"
    
    def _refine_strategy(self, state: AgentState) -> AgentState:
        """Refine search strategy based on current results"""
        print("🔧 Refining search strategy...")
        
        # Analyze what's working and what's not
        refinement_prompt = ChatPromptTemplate.from_template("""
        Based on the current progress, suggest new search strategies:
        
        Goal: {goal}
        Target: {target_count} items
        Current results: {current_count} items
        Tried queries: {tried_queries}
        
        What new approaches, keywords, or sources should we try?
        Return 5 new search queries as a JSON list.
        """)
        
        response = self.llm.invoke(refinement_prompt.format(
            goal=state.goal,
            target_count=state.target_count,
            current_count=len(state.current_results),
            tried_queries=state.attempted_sources
        ))
        
        try:
            new_queries = json.loads(response.content)
            for query in new_queries:
                if query not in state.search_queries:
                    state.search_queries.append(query)
        except:
            # Generate fallback queries
            base_terms = state.goal.split()
            for term in base_terms:
                new_query = f"best {term} resources 2024"
                if new_query not in state.search_queries:
                    state.search_queries.append(new_query)
        
        return state
    
    def _finalize_results(self, state: AgentState) -> AgentState:
        """Finalize and format the results"""
        print("📋 Finalizing results...")
        
        # Sort and format results
        state.current_results = state.current_results[:state.target_count]
        
        final_message = f"""
🎯 Goal: {state.goal}
✅ Status: {state.status.value.upper()}
📊 Results: {len(state.current_results)}/{state.target_count}
🔄 Iterations: {state.iteration_count}

Results:
"""
        
        for i, result in enumerate(state.current_results, 1):
            if 'url' in result and result['url']:
                final_message += f"{i}. {result.get('title', 'No title')} - {result.get('url')}\n"
                if result.get('description'):
                    final_message += f"   Description: {result.get('description')}\n"
                if result.get('emails'):
                    final_message += f"   Emails: {', '.join(result.get('emails'))}\n"
            elif 'email' in result and result['email']:
                final_message += f"{i}. {result.get('name', 'Unknown')} - {result.get('email')} ({result.get('company', 'Unknown Company')})\n"
                if result.get('role'):
                    final_message += f"   Role: {result.get('role')}\n"
            elif 'contact_info' in result:
                final_message += f"{i}. Contact Information: {', '.join(result.get('contact_info'))}\n"
            else:
                final_message += f"{i}. {result.get('title', 'No title')}\n"
                if result.get('description'):
                    final_message += f"   {result.get('description')}\n"
        
        state.messages.append(AIMessage(content=final_message))
        print(final_message)
        return state
    
    async def run_agent(self, goal: str, max_iterations: int = 5) -> AgentState:
        """Run the persistent agent until goal is achieved"""
        print(f"🚀 Starting persistent agent for goal: {goal}")
        
        # Initialize state
        initial_state = AgentState(
            goal=goal,
            target_count=10,  # Will be updated in analyze_goal
            current_results=[],
            search_queries=[],
            attempted_sources=[],
            status=GoalStatus.PENDING,
            iteration_count=0,
            max_iterations=max_iterations,
            messages=[HumanMessage(content=goal)]
        )
        
        # Run the graph
        config = {"configurable": {"thread_id": f"agent_{hash(goal)}"}}
        
        # Use astream to handle async execution properly
        final_state = None
        async for state in self.graph.astream(initial_state, config):
            final_state = state
            # Print progress
            for key, value in state.items():
                if hasattr(value, 'status'):
                    print(f"Node {key}: Status = {value.status.value}")

        # If we have a final state, extract the actual AgentState
        if final_state:
            # Get the last state from the stream
            last_node_key = list(final_state.keys())[-1]
            final_agent_state = final_state[last_node_key]
            
            # Convert dict to AgentState if needed
            if isinstance(final_agent_state, dict):
                status = final_agent_state.get("status")
                if isinstance(status, str):
                    status = GoalStatus(status)
                final_agent_state = AgentState(
                    goal=final_agent_state.get("goal"),
                    target_count=final_agent_state.get("target_count"),
                    current_results=final_agent_state.get("current_results"),
                    search_queries=final_agent_state.get("search_queries"),
                    attempted_sources=final_agent_state.get("attempted_sources"),
                    status=status,
                    iteration_count=final_agent_state.get("iteration_count"),
                    max_iterations=final_agent_state.get("max_iterations"),
                    messages=final_agent_state.get("messages"),
                    last_error=final_agent_state.get("last_error")
                )
            
            return final_agent_state
        
        # Fallback if no final state
        return initial_state

# Example usage
async def main():
    # Replace with your Groq API key
    GROQ_API_KEY = ""
    
    agent = PersistentSearchAgent(GROQ_API_KEY)
    
    # Example goals with enhanced web search
    goals = [
        "Find 10 email addresses of recruiters who are actively hiring AI/ML roles",
        "Collect 10 startup founder contact emails in fintech sector"
    ]
    
    for goal in goals:
        print(f"\n{'='*50}")
        print(f"GOAL: {goal}")
        print(f"{'='*50}")
        
        try:
            result = await agent.run_agent(goal, max_iterations=5)
            print(f"\n✅ Final Status: {result.status.value}")
            print(f"📊 Results Found: {len(result.current_results)}")
            
            # Show sample results
            if result.current_results:
                print("\n🔍 Sample Results:")
                for i, res in enumerate(result.current_results[:3], 1):
                    print(f"{i}. {res.get('title', 'No title')}")
                    if res.get('url'):
                        print(f"🔗 URL: {res.get('url')}")
                    if res.get('emails'):
                        print(f"📨 Emails: {', '.join(res.get('emails'))}")
                    if res.get('description'):
                        print(f"🗒️ Description: {res.get('description')[:100]}...")
                    print()
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Wait between goals
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())