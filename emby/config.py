class Logging:
    """Logging

    Members
    --------

    Nothing
        No logging what-so-ever
    Progress
        Logging of progress such as training etc
    Everything
        Logging of everything, as verbose as possible




    """
    Nothing = 0
    Progress = 1
    Everything = 2


class Device:
    """Device

    Members
    -------

    Detect
        Picks GPU if available, otherwise CPU
    CPU
        Picks CPU (or throws error)
    GPU
        Picks GPU (or throws error)

    """

    Detect = 0
    CPU = 1
    GPU = 2
