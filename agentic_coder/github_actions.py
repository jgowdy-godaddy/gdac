from __future__ import annotations
import os
import yaml
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

"""
GitHub Actions Integration
Provides automation capabilities for CI/CD workflows similar to Claude Code.
"""

class GitHubActionsManager:
    """Manage GitHub Actions workflows for automated tasks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.workflows_dir = os.path.join(repo_path, ".github", "workflows")
        self.ensure_workflows_dir()
    
    def ensure_workflows_dir(self):
        """Ensure the workflows directory exists."""
        os.makedirs(self.workflows_dir, exist_ok=True)
    
    def create_code_review_workflow(self) -> str:
        """Create a workflow for automated code review on PRs."""
        workflow = {
            'name': 'Agentic Code Review',
            'on': {
                'pull_request': {
                    'types': ['opened', 'synchronize']
                }
            },
            'jobs': {
                'review': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4',
                            'with': {
                                'fetch-depth': 0  # Fetch all history for diff
                            }
                        },
                        {
                            'name': 'Setup Python',
                            'uses': 'actions/setup-python@v5',
                            'with': {
                                'python-version': '3.12'
                            }
                        },
                        {
                            'name': 'Install Agentic Coder',
                            'run': 'pip install -e .'
                        },
                        {
                            'name': 'Run Code Review',
                            'env': {
                                'GITHUB_TOKEN': '${{ secrets.GITHUB_TOKEN }}',
                                'OPENAI_API_KEY': '${{ secrets.OPENAI_API_KEY }}'
                            },
                            'run': '''
                                # Get the diff
                                git diff origin/${{ github.base_ref }}..HEAD > changes.diff
                                
                                # Run agentic review
                                python -c "
from agentic_coder.runtime import Agent
from agentic_coder.tools import ToolRegistry

agent = Agent('remote-openai', '.')
tools = ToolRegistry('.')

# Read the diff
with open('changes.diff', 'r') as f:
    diff = f.read()

# Analyze the code
goal = f'Review this code diff and provide feedback:\\n{diff[:5000]}'
agent.run(goal, max_iters=5)
"
                            '''
                        },
                        {
                            'name': 'Post Review Comment',
                            'if': 'always()',
                            'uses': 'actions/github-script@v7',
                            'with': {
                                'script': '''
                                    const fs = require('fs');
                                    const review = fs.readFileSync('review.md', 'utf8');
                                    
                                    await github.rest.pulls.createReview({
                                        owner: context.repo.owner,
                                        repo: context.repo.repo,
                                        pull_number: context.issue.number,
                                        body: review,
                                        event: 'COMMENT'
                                    });
                                '''
                            }
                        }
                    ]
                }
            }
        }
        
        workflow_path = os.path.join(self.workflows_dir, "agentic_code_review.yml")
        with open(workflow_path, 'w') as f:
            yaml.dump(workflow, f, default_flow_style=False)
        
        return workflow_path
    
    def create_test_automation_workflow(self) -> str:
        """Create a workflow for automated testing."""
        workflow = {
            'name': 'Agentic Test Runner',
            'on': {
                'push': {
                    'branches': ['main', 'develop']
                },
                'pull_request': {
                    'branches': ['main']
                }
            },
            'jobs': {
                'test': {
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.10', '3.11', '3.12']
                        }
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Setup Python',
                            'uses': 'actions/setup-python@v5',
                            'with': {
                                'python-version': '${{ matrix.python-version }}'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '''
                                pip install -e .
                                pip install pytest pytest-cov
                            '''
                        },
                        {
                            'name': 'Run tests with coverage',
                            'run': 'pytest --cov=agentic_coder --cov-report=xml --cov-report=term'
                        },
                        {
                            'name': 'Upload coverage',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {
                                'file': './coverage.xml',
                                'fail_ci_if_error': False
                            }
                        },
                        {
                            'name': 'Run linting',
                            'run': '''
                                pip install ruff black mypy
                                ruff check agentic_coder/
                                black --check agentic_coder/
                                mypy agentic_coder/ --ignore-missing-imports
                            '''
                        }
                    ]
                }
            }
        }
        
        workflow_path = os.path.join(self.workflows_dir, "agentic_test_runner.yml")
        with open(workflow_path, 'w') as f:
            yaml.dump(workflow, f, default_flow_style=False)
        
        return workflow_path
    
    def create_documentation_workflow(self) -> str:
        """Create a workflow for automated documentation generation."""
        workflow = {
            'name': 'Agentic Documentation',
            'on': {
                'push': {
                    'branches': ['main'],
                    'paths': ['**.py', 'docs/**', 'README.md']
                }
            },
            'jobs': {
                'docs': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Setup Python',
                            'uses': 'actions/setup-python@v5',
                            'with': {
                                'python-version': '3.12'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': '''
                                pip install -e .
                                pip install sphinx sphinx-rtd-theme autodoc
                            '''
                        },
                        {
                            'name': 'Generate API docs',
                            'run': '''
                                sphinx-apidoc -o docs/api agentic_coder/
                                cd docs && make html
                            '''
                        },
                        {
                            'name': 'Deploy to GitHub Pages',
                            'uses': 'peaceiris/actions-gh-pages@v3',
                            'if': "github.ref == 'refs/heads/main'",
                            'with': {
                                'github_token': '${{ secrets.GITHUB_TOKEN }}',
                                'publish_dir': './docs/_build/html'
                            }
                        }
                    ]
                }
            }
        }
        
        workflow_path = os.path.join(self.workflows_dir, "agentic_documentation.yml")
        with open(workflow_path, 'w') as f:
            yaml.dump(workflow, f, default_flow_style=False)
        
        return workflow_path
    
    def create_release_workflow(self) -> str:
        """Create a workflow for automated releases."""
        workflow = {
            'name': 'Agentic Release',
            'on': {
                'push': {
                    'tags': ['v*']
                }
            },
            'jobs': {
                'release': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Setup Python',
                            'uses': 'actions/setup-python@v5',
                            'with': {
                                'python-version': '3.12'
                            }
                        },
                        {
                            'name': 'Install build tools',
                            'run': 'pip install build twine'
                        },
                        {
                            'name': 'Build package',
                            'run': 'python -m build'
                        },
                        {
                            'name': 'Create GitHub Release',
                            'uses': 'softprops/action-gh-release@v1',
                            'with': {
                                'files': 'dist/*',
                                'generate_release_notes': True
                            },
                            'env': {
                                'GITHUB_TOKEN': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Publish to PyPI',
                            'if': 'startsWith(github.ref, \'refs/tags/\')',
                            'env': {
                                'TWINE_USERNAME': '__token__',
                                'TWINE_PASSWORD': '${{ secrets.PYPI_API_TOKEN }}'
                            },
                            'run': 'twine upload dist/*'
                        }
                    ]
                }
            }
        }
        
        workflow_path = os.path.join(self.workflows_dir, "agentic_release.yml")
        with open(workflow_path, 'w') as f:
            yaml.dump(workflow, f, default_flow_style=False)
        
        return workflow_path
    
    def create_security_scan_workflow(self) -> str:
        """Create a workflow for security scanning."""
        workflow = {
            'name': 'Security Scan',
            'on': {
                'push': {
                    'branches': ['main']
                },
                'pull_request': {
                    'branches': ['main']
                },
                'schedule': [
                    {'cron': '0 0 * * 1'}  # Weekly on Monday
                ]
            },
            'jobs': {
                'security': {
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Setup Python',
                            'uses': 'actions/setup-python@v5',
                            'with': {
                                'python-version': '3.12'
                            }
                        },
                        {
                            'name': 'Install security tools',
                            'run': '''
                                pip install safety bandit semgrep
                            '''
                        },
                        {
                            'name': 'Run safety check',
                            'run': 'safety check --json > safety-report.json',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Run bandit scan',
                            'run': 'bandit -r agentic_coder/ -f json -o bandit-report.json',
                            'continue-on-error': True
                        },
                        {
                            'name': 'Run Semgrep',
                            'uses': 'returntocorp/semgrep-action@v1',
                            'with': {
                                'config': 'auto'
                            }
                        },
                        {
                            'name': 'Upload security reports',
                            'uses': 'actions/upload-artifact@v3',
                            'if': 'always()',
                            'with': {
                                'name': 'security-reports',
                                'path': '*-report.json'
                            }
                        }
                    ]
                }
            }
        }
        
        workflow_path = os.path.join(self.workflows_dir, "security_scan.yml")
        with open(workflow_path, 'w') as f:
            yaml.dump(workflow, f, default_flow_style=False)
        
        return workflow_path
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflow files."""
        workflows = []
        
        if not os.path.exists(self.workflows_dir):
            return workflows
        
        for filename in os.listdir(self.workflows_dir):
            if filename.endswith(('.yml', '.yaml')):
                workflow_path = os.path.join(self.workflows_dir, filename)
                try:
                    with open(workflow_path, 'r') as f:
                        workflow_data = yaml.safe_load(f)
                    
                    workflows.append({
                        'file': filename,
                        'name': workflow_data.get('name', 'Unnamed'),
                        'triggers': list(workflow_data.get('on', {}).keys()),
                        'jobs': list(workflow_data.get('jobs', {}).keys())
                    })
                except Exception as e:
                    workflows.append({
                        'file': filename,
                        'name': 'Error reading workflow',
                        'error': str(e)
                    })
        
        return workflows
    
    def validate_workflow(self, workflow_file: str) -> Dict[str, Any]:
        """Validate a workflow file."""
        workflow_path = os.path.join(self.workflows_dir, workflow_file)
        
        if not os.path.exists(workflow_path):
            return {'valid': False, 'error': 'Workflow file not found'}
        
        try:
            with open(workflow_path, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            # Basic validation
            errors = []
            if 'name' not in workflow_data:
                errors.append('Missing workflow name')
            if 'on' not in workflow_data:
                errors.append('Missing trigger events')
            if 'jobs' not in workflow_data or not workflow_data['jobs']:
                errors.append('No jobs defined')
            
            # Validate job structure
            for job_name, job_config in workflow_data.get('jobs', {}).items():
                if 'runs-on' not in job_config:
                    errors.append(f"Job '{job_name}' missing 'runs-on'")
                if 'steps' not in job_config or not job_config['steps']:
                    errors.append(f"Job '{job_name}' has no steps")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'workflow': workflow_data
            }
            
        except yaml.YAMLError as e:
            return {'valid': False, 'error': f'YAML parsing error: {e}'}
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def create_all_workflows(self) -> List[str]:
        """Create all standard workflows."""
        created = []
        
        try:
            created.append(self.create_code_review_workflow())
            created.append(self.create_test_automation_workflow())
            created.append(self.create_documentation_workflow())
            created.append(self.create_release_workflow())
            created.append(self.create_security_scan_workflow())
        except Exception as e:
            print(f"Error creating workflows: {e}")
        
        return created