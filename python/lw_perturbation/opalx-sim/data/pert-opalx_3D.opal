
OPTION, PSDUMPFREQ      = 10000;
OPTION, STATDUMPFREQ    = 1;
OPTION, BOUNDPDESTROY   = 10;
OPTION, AUTOPHASE       = 4;
OPTION, VERSION         = 10900;
OPTION, STEPINFOFQ      = 20;

Title, string="LW perturbation benchmark: Gaussian bunch in constant Ez";

Value,{OPALVERSION};

REAL n_particles         = 1E6;
REAL beam_bunch_charge   = 1e-9;
          REAL rf_freq             = 1.3e9;
         
REAL sigma_x             = 1.0e-3;
        REAL sigma_y             = 1.0e-3;
        REAL sigma_z             = 1.0e-1;
        
REAL p_rms_MeVc          = 1.0e-6;
        REAL p_rms_betagamma     = p_rms_MeVc / 0.511;

REAL t_end               = 15.0e-9;
       REAL n_steps             = 1000;
REAL dt                  = t_end / n_steps;

REAL Edes    = 1e-9;
           REAL gamma   = (Edes+EMASS)/EMASS;
 REAL beta    = sqrt(1-(1/gamma^2));
REAL P0      = gamma*beta*EMASS;

Value,{Edes, P0, p_rms_betagamma, dt};

\1Z = -1.0\2    EZ       = -1.0;

myLine: Line = (E1);

Dist1_GAUSS: DISTRIBUTION, TYPE=GAUSS,    SIGMAX    = sigma_x,    SIGMAY    = sigma_y,    SIGMAZ    = sigma_z,    SIGMAPX   = p_rms_betagamma,    SIGMAPY   = p_rms_betagamma,    SIGMAPZ   = p_rms_betagamma,    NPARTDIST = n_particles;

ES1: EMISSIONSOURCE, DISTRIBUTION = Dist1_GAUSS;
mySources: EMISSIONSOURCELIST = (ES1);

BEAM1: BEAM,    PARTICLE = ELECTRON,    NALLOC   = n_particles,    BCHARGE  = beam_bunch_charge,    SOURCES  = mySources,    CHARGE   = -1,    pc       = P0;

BINS1: BINNING,    MAXBINS        = 120,    DESIREDWIDTH   = 0.05,    TABLEPRINTFREQ = 40,    PARAMETER      = GAMMAZ;

FS1: Fieldsolver, TYPE=OPEN, BINS=BINS1,    NX        = 64,    NY        = 64,    NZ        = 64,    PARFFTX   = true,    PARFFTY   = true,    PARFFTZ   = false,    BCFFTX    = OPEN,    BCFFTY    = OPEN,    BCFFTZ    = OPEN, BBOXINCR = 5, GREENSF = STANDARD;

TRACK, LINE = myLine, BEAM = BEAM1, MAXSTEPS = n_steps, DT = dt, ZSTOP = 4.5;
RUN, METHOD = "PARALLEL", FIELDSOLVER = FS1;
ENDTRACK;
QUIT;

