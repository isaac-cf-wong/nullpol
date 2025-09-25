# Contributing to nullpol

Thank you for considering contributing to nullpol! We welcome contributions from the community and are excited to work with you. To ensure a smooth contribution process, please follow the guidelines below.

## How to Contribute
### 1. Getting Started
#### 1. Fork the Repository:
- Click the "Fork" button at the top right of this page to create a copy of this repository under your Git account.

#### 2. Clone Your Fork:
- Clone the forked repository to your local machine:
```bash
git clone git@git.ligo.org:bayesian-null-stream/nullpol.git
```
- Navigate to the repository
```bash
cd nullpol
```

#### 3. Create an issue
- Create an issue to explain the proposal of the code changes
- Depending on the nature of the proposal, specify it as a prefix in the title with the format: "<prefix>: <description>"
  - A new feature: feature
  - A bugfix: bugfix
  - An optimization of the codes: optimize

#### 3. Create a New Branch:
- Create a new branch for you changes:
```bash
git checkout -b <id of the issue>-<prefix>-<description>
```

### 2. Making Changes
#### 1. Make Your Changes:
- Implement your changes or new features in your branch.

#### 2. Write Tests:
- If applicable, write tests to cover your changes. Ensure that all tests pass.

#### 3. Update Documentation:
- Update the documentation if your changes affect it.

#### 4. Commit Your Changes:
- Commit your changes with a meaningful commit message:
```bash
git add .
git commit -m "Add a descriptive commit message"
```

### 3. Submitting Your Changes
#### 1. Push Your Branch:
- Push your changes to your forked repository:
```bash
git push origin feature/your-feature-name
```

#### 2. Open a Pull Request:
- Go to the Pull Requests tab on the main repository.
- Click "New Pull Request".
- Select the branch you created and click "Create Pull Request".
- Provide a descriptive title and detailed description of your changes.

#### 4. Review Process
- Review: Your pull request will be reviewed by the maintainers. Be prepared to respond to feedback and make additional changes if necessary.
- Approval: Once approved, your changes will be merged into the release branch.

## Code of Conduct
Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a welcoming and respectful environment for all contributors.

## Reporting Issues
If you encounter any issues or bugs, please [report them](https://git.ligo.org/bayesian-null-stream/nullpol/-/issues) on our Git Issues page. Include relevant details and steps to reproduce the issue.

## Contact
For any questions or further assistance, please open an issue on Git.

## Thank You!

We appreciate your contributions to nullpol and look forward to your pull requests!
