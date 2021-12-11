from ...algos.egrl.learner import Learner


def initialize_portfolio(portfolio, args, genealogy, portfolio_id, model_constructor):
	"""Portfolio of learners

        Parameters:
            portfolio (list): Incoming list
            args (object): param class

        Returns:
            portfolio (list): Portfolio of learners
    """



	if portfolio_id == 10:
		# Learner 1
		portfolio.append(
			Learner(model_constructor, args, gamma=0.9))

		# Learner 2
		portfolio.append(
			Learner(model_constructor, args, gamma=0.99))

		# Learner 3
		portfolio.append(
			Learner(model_constructor, args, gamma=0.999))


	elif portfolio_id == 1:
		# Learner 1
		portfolio.append(
			Learner(model_constructor, args, gamma=0.99))

	else:
		raise("Incorrect Portfolio choice")




	return portfolio
