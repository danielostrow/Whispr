# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker support for containerized execution
- GitHub Actions workflows for CI/CD:
  - Python Tests workflow for multi-platform testing
  - Python Lint workflow for code quality
  - C Extensions Build workflow for testing C extensions
  - Integration Tests workflow for end-to-end testing
  - Changelog Checker workflow to enforce changelog updates
  - Docker Build and Test workflow
  - CodeQL Analysis workflow for security scanning
  - Automatic Changelog Update workflow
- Added Docker usage instructions to README.md
- Added GitHub Actions workflow badges to README.md
- Added PR title formatting guidelines for automatic changelog updates

### Changed
- Updated run.sh to support Docker execution with -d/--docker flag
- Restructured README.md to include Docker and CI information
- Updated GitHub Actions workflows to use actions/upload-artifact@v4 (from deprecated v2)
- Updated GitHub Actions to use actions/setup-python@v5 (from v4)
- Updated audio sample URLs to use reliable sources

### Fixed
- Fixed failing CI workflows that were using deprecated GitHub Actions
- Fixed timeout handling in integration tests on macOS
- Added proper error handling for C extensions build process
- Added permissions for Docker build workflow
- Added conditional logic to skip CMake steps if CMakeLists.txt doesn't exist
- Updated codecov action to v4 and added token handling

## [0.1.0] - 2025-06-11

### Added
- Initial release with core functionality
- Voice Activity Detection (VAD)
- Speaker Clustering
- Basic source separation
- Interactive UI with Dash
- C extensions for improved performance 