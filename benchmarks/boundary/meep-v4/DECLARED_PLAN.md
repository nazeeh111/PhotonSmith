# Independent Meep time-domain validation plan

Declared before any new Meep run. Actual final geometry of the existing1.73um/index2.15/target-1 inverse-design study, not the skill helper's oldaccepted1550nmcase. PhysicalJSONdesign.json and native-reference.jsonpin exactinputsandthe source. Nativeorders21/31/45/61werechecked; M45→M61maxopenpowerchange3.13e-6<=1e-4. M61referencewillbecomparedwithMeep1.35.0. No fittedphase orrescaling.

Nine stages:exactvacuumanduniformslabcontrols,freshdeviceat40/80/160pixelsperum,80pixelstime250→500,PML1→2,air1.5→3,andglobaltranslation.137period. Everyrunhasitsownmatchedemptyreferenceandcomplexfieldsubtraction. Nativepowerlimit.005matchesinitialfinishline;grid80→160difference<=.0025isstricterthanits.005spatialcomparisonlimit. Otherfrozenhelperdefaultsapplyincludingenergybalance,fluxprojection,time/PML/paddinganddeembeddedcomplexphase. KeeprawNPZarrays,allfailedattemptsandjournal. No clipping,forcingconservation,unboundedretriesorpost-resultthresholdweakening.

Budget3600secondsfornine-stagerun,withpost-source time andsourcepulse recorded byhelper. Genuinefreeownedserialruntime,OMP/BLAS1,isolatedimports,-B,noload overrides,tasklocalMatplotlibcache. Launchonlyafterv4timingscompleted. NumericalMeepagreementdoesnotcountasindependenthumanreview,physicaldevicevalidationoradoption.
