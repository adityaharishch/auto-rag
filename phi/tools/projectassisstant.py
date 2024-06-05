import json
from phi.tools import Toolkit
from phi.utils.log import logger

try:
    from jira import JIRA
except ImportError:
    raise ImportError("`jira` library is not installed. Please install it using `pip install jira`.")


# Create a JIRA client instance
jira_client = JIRA(server=JIRA_URL, basic_auth=(JIRA_USERNAME, JIRA_API_TOKEN))

class ProjectTools(Toolkit):
    def __init__(
        self,
        project_updates: bool = True,
        milestone_tracking: bool = True,
        budget_analysis: bool = True,
        resource_management: bool = True,
        risk_analysis: bool = True,
        jira_integration: bool = True,
    ):
        super().__init__(name="project_tools")

        if project_updates:
            self.register(self.get_project_updates)
        if milestone_tracking:
            self.register(self.get_milestone_tracking)
        if budget_analysis:
            self.register(self.get_budget_analysis)
        if resource_management:
            self.register(self.get_resource_management)
        if risk_analysis:
            self.register(self.get_risk_analysis)
        if jira_integration:
            self.register(self.get_jira_issues)

    def get_project_updates(self, project_id: str) -> str:
        """Retrieve latest updates and status for a given project."""
        try:
            project = jira_client.project(project_id)
            updates = {
                "Project Name": project.name,
                "Project Key": project.key,
                "Project Lead": project.lead.displayName,
                "Project URL": project.self,
            }
            return json.dumps(updates, indent=2)
        except Exception as e:
            return f"Error fetching updates for project {project_id}: {e}"

    def get_milestone_tracking(self, project_id: str) -> str:
        """Get the current status of project milestones."""
        try:
            issues = jira_client.search_issues(f'project={project_id} AND issuetype="Milestone"')
            milestones = [{issue.key: issue.fields.summary} for issue in issues]
            return json.dumps(milestones, indent=2)
        except Exception as e:
            return f"Error tracking milestones for project {project_id}: {e}"

    def get_budget_analysis(self, project_id: str) -> str:
        """Provide budget expenditure and remaining budget analysis."""
        # Implement budget analysis based on custom fields or integration with financial tools
        return "Budget analysis functionality needs implementation."

    def get_resource_management(self, project_id: str) -> str:
        """Retrieve information about resource allocation and utilization."""
        # Implement resource management analysis based on project data
        return "Resource management functionality needs implementation."

    def get_risk_analysis(self, project_id: str) -> str:
        """Analyze potential risks and their current management strategies."""
        try:
            risks = jira_client.search_issues(f'project={project_id} AND issuetype="Risk"')
            risk_details = [{risk.key: risk.fields.summary} for risk in risks]
            return json.dumps(risk_details, indent=2)
        except Exception as e:
            return f"Error analyzing risks for project {project_id}: {e}"

    def get_jira_issues(self, project_id: str, issue_type: str = "All") -> str:
        """Fetch issues from JIRA based on issue type."""
        try:
            jql_query = f'project={project_id}'
            if issue_type != "All":
                jql_query += f' AND issuetype="{issue_type}"'
            issues = jira_client.search_issues(jql_query)
            issue_details = [{issue.key: issue.fields.summary} for issue in issues]
            return json.dumps(issue_details, indent=2)
        except Exception as e:
            return f"Error fetching JIRA issues for {project_id}: {e}"

