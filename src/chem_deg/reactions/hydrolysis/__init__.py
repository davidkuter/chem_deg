from chem_deg.reactions.hydrolysis.anhydride import (
    AnhydrideHydrolysisAcyclic,
    AnhydrideHydrolysisCyclicFive,
    AnhydrideHydrolysisCyclicSix,
)
from chem_deg.reactions.hydrolysis.carbamate import (
    CarbamateHydrolysis,
    ThiocarbamateHydrolysis,
)
from chem_deg.reactions.hydrolysis.carbonate import (
    CarbonateHydrolysisAcyclic,
    CarbonateHydrolysisCyclic,
)
from chem_deg.reactions.hydrolysis.ester import (
    CarboxylateEsterHydrolysis,
    PhosphorusEsterHydrolysisAcid,
    PhosphorusEsterHydrolysisBase,
)
from chem_deg.reactions.hydrolysis.halogenated_aliphatic import (
    HalogenatedAliphaticsSubstitutionA,
    HalogenatedAliphaticsSubstitutionC,
    HalogenatedAliphaticsElimination,
)
from chem_deg.reactions.hydrolysis.imide import (
    ImideHydrolysisFive,
    ImideHydrolysisSix,
)
from chem_deg.reactions.hydrolysis.lactam import (
    LactamHydrolysisFour,
    LactamHydrolysisFive,
    LactamHydrolysisSix,
)
from chem_deg.reactions.hydrolysis.lactone import (
    LactoneHydrolysisFour,
    LactoneHydrolysisFive,
    LactoneHydrolysisSix,
)
from chem_deg.reactions.hydrolysis.misc import (
    EpoxideHydrolysis,
    AmideHydrolysis,
    NitrileHydrolysis,
    NSHydrolysis,
    AcidHalideHydrolysis,
)
from chem_deg.reactions.hydrolysis.urea import (
    UreaHydrolysisAcyclic,
    UreaHydrolysisCyclicFive,
    UreaHydrolysisCyclicSix,
    SulfonylureaHydrolysis,
)

hydrolysis_reactions = [
    HalogenatedAliphaticsSubstitutionA(),
    HalogenatedAliphaticsSubstitutionC(),
    HalogenatedAliphaticsElimination(),
    EpoxideHydrolysis(),
    PhosphorusEsterHydrolysisAcid(),
    PhosphorusEsterHydrolysisBase(),
    CarboxylateEsterHydrolysis(),
    LactoneHydrolysisFour(),
    LactoneHydrolysisFive(),
    LactoneHydrolysisSix(),
    CarbonateHydrolysisAcyclic(),
    CarbonateHydrolysisCyclic(),
    AnhydrideHydrolysisAcyclic(),
    AnhydrideHydrolysisCyclicFive(),
    AnhydrideHydrolysisCyclicSix(),
    AmideHydrolysis(),
    LactamHydrolysisFour(),
    LactamHydrolysisFive(),
    LactamHydrolysisSix(),
    CarbamateHydrolysis(),
    ThiocarbamateHydrolysis(),
    UreaHydrolysisAcyclic(),
    UreaHydrolysisCyclicFive(),
    UreaHydrolysisCyclicSix(),
    SulfonylureaHydrolysis(),
    NitrileHydrolysis(),
    NSHydrolysis(),
    ImideHydrolysisFive(),
    ImideHydrolysisSix(),
    AcidHalideHydrolysis(),
]
