from chem_deg.reactions.base import ReactionClass
from chem_deg.reactions.hydrolysis import hydrolysis_reactions


class Hydrolysis(ReactionClass):
    """
    Hydrolysis Reaction Class.
    Accesses all hydrolysis reactions.
    """

    def __init__(self):
        super().__init__(
            name="Hydrolysis",
            reactions=hydrolysis_reactions,
        )
