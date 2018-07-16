In this project, I have worked with a company's email network where each node corresponds to a person at the company, and each edge indicates that at least one email has been sent between two people.
The network also contains the node attributes Department and ManagementSalary.
Department indicates the department in the company which the person belongs to, and ManagementSalary indicates whether that person is receiving a management position salary.
Using network G, I have identified the people in the network with missing values for the node attribute ManagementSalary and predicted whether or not these individuals are receiving a management position salary.
To accomplish this, I have created a matrix of node features using networkx, trained a Logistic Regression classifier on nodes that have ManagementSalary data, and predict a probability of the node receiving a management salary for nodes where ManagementSalary is missing.
The predictions have been given as the probability that the corresponding employee is receiving a management position salary.
