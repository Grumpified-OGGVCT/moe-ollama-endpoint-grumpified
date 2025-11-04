# Contributing to MoE Ollama Endpoint

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/moe-ollama-endpoint-grumpified.git`
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Make your changes
5. Test your changes thoroughly
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black ruff mypy

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run tests
pytest

# Format code
black app/ tests/

# Lint
ruff check app/ tests/

# Type check
mypy app/
```

## Pull Request Process

1. **Update tests**: Add tests for any new functionality
2. **Update documentation**: Update README.md, API.md, or DEVELOPMENT.md as needed
3. **Code quality**: Ensure all checks pass (tests, linting, formatting)
4. **Clear description**: Explain what changes you made and why
5. **Small PRs**: Keep PRs focused on a single feature or fix

## Code Style

- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Use meaningful variable names

### Example:

```python
async def process_request(
    query: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a user query and return a response.
    
    Args:
        query: The user's query string
        model: Optional model name to use
    
    Returns:
        Dictionary containing the response data
    """
    # Implementation here
    pass
```

## Testing

- Write tests for all new features
- Ensure existing tests still pass
- Aim for >80% code coverage
- Test edge cases and error conditions

### Running Tests:

```bash
# All tests
pytest

# Specific file
pytest tests/test_api.py

# With coverage
pytest --cov=app --cov-report=html
```

## Commit Messages

Use clear, descriptive commit messages:

```
Add RAG search endpoint with filtering

- Implement vector similarity search
- Add collection filtering
- Update API documentation
- Add tests for new endpoint
```

## Feature Requests

Open an issue with:
- Clear description of the feature
- Use case and benefits
- Proposed implementation (if you have one)

## Bug Reports

Open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and logs

## Areas for Contribution

### High Priority
- Performance optimizations
- Additional model routing strategies
- Enhanced RAG capabilities
- Better error handling
- Security improvements

### Medium Priority
- Additional API endpoints
- Improved documentation
- More examples
- Integration tests
- Monitoring and metrics

### Nice to Have
- UI dashboard
- CLI tools
- Additional model providers
- Caching layer
- Rate limiting

## Questions?

If you have questions, feel free to:
- Open a discussion
- Open an issue with the "question" label
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be recognized in:
- Release notes
- CONTRIBUTORS.md file
- Project documentation

Thank you for contributing! 🎉
