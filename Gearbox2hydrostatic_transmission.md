

# The note to modify OpenFAST source files (HWT)

This note is used to record the modification process replacing the gearbox drivetrain with the hydraulic drivetrain in `OpenFAST`.

OpenFAST uses the gear-box based drivetrain for energy conversion, and in this paper, we replace the gear-box based drivetrain with the hydrostatic drivetrain by modifying the ServoDyn module in OpenFAST. First, the drivetrain rotational-flexibility DOF is closed in the ElastoDyn input file (.dat) and the GBRatio is set to 1. Then, we regard the generator in GWT as the hydraulic pump in HWT and modify its inertial in the FAST input file (.fst). Finally, the transmission dynamics of hydraulic system in HWT is modelled as a function in the ServoDyn module and it will be called before the state update of servo system. Finally, we embed the RL-based control policy in the UserVSCont_KP.f90 file.

## Parameters
### Genereal parameters

| Name                                                         | Sign       | Unit               | Value                |
| ------------------------------------------------------------ | ---------- | ------------------ | -------------------- |
| Moments of inertia of the pump                               | $J_p$      | $kg \cdot m^2$     | 3680                 |
| Moments of inertia of the motor                              | $J_m$      | $kg \cdot m^2$     | 50                   |
| Pump displacement                                            | $D_p$      | $L/rev$​            | 626                  |
| Motor displacement                                           | $D_m$      | $L/rev$            | 4.9                  |
| Viscous damping of pump                                      | $B_p$      | $N\cdot m \cdot s$ | $50\times 10^3$      |
| Viscous damping of motor                                     | $B_m$      | $N\cdot m \cdot s$ | 2.5                  |
| Coulomb friction coefficient of the pump                     | $C_{fp}$   | -                  | 0.02                 |
| Coulomb friction coefficient of the motor                    | $C_{fm}$   | -                  | 0.02                 |
| Laminar leakage coefficients of the pump                     | $C_{sp}$   | $m^3/s/Pa$         | $7.1\times 10^{-11}$ |
| Laminar leakage coefficients of the motor                    | $C_{sm}$   | $m^3/s/Pa$         | $7.0\times 10^{-11}$ |
| The fixed rotational speed (motor and synchronous generator) | $\omega_m$ | $rad/s$            | $2\pi*60(f)$         |

### Parameter of tube line

| Name                            | Sign   | Unit           | Value              |
| ------------------------------- | ------ | -------------- | ------------------ |
| High pressure oil line length   | $L$    | $m$            | 100                |
| Oil pipe line internal diameter | $r$    | $m$            | 0.25               |
| Density of mineral oil          | $\rho$ | $kg \cdot m^3$ | 917                |
| Kinematic viscosity of oil      | $\nu$  | $m^2/s$        | $40\times 10^{-6}$ |
| Effective bulk modulus of oil   | $E$    | $Pa$           | $1.0\times 10^9$   |

## Pre-modification

1. Change to the file of `/Users/huangyubo/openfast/reg_tests/r-test/glue-codes/fast-farm/TSinflow/NRELOffshrBsline5MW_Onshore_ElastoDyn_8mps.dat`

2. close the drivetrain DOF 

   ```fortran
   False	      DrTrDOF     - Drivetrain rotational-flexibility DOF (flag)
   ```

   

3. Regard the generator DOF as the Pump DOF and set $n=1$ (Gearbox ratio)

   ```fortran
   3680   GenIner     - Generator inertia about HSS (kg m^2)
   1   GBRatio     - Gearbox ratio (-)
   ```

   

## Define the state variables of the hydralic line 

The state variables of the hydralic line used in the calculation are: 

$$P_p, P_m, Q_p, Q_m, HL_{state}$$

Where $HL_{state}$ is the state vector $x$ in:

$$P = C \times x$$

~~These variables will be created in the file name`ElastoDyn_Registry.txt`(537-541), locating at:~~

```fortran
# ..... States .....
...............................................................................................................
# Define continuous (differentiable) states here:
```

~~The creating code is:~~

```fortran
typedef	^	ContinuousStateType	ReKi	P_p	-	-	-	"Current pressure difference across the pump"
typedef	^	ContinuousStateType	ReKi	P_m	-	-	-	"Current pressure difference across the moter"
typedef	^	ContinuousStateType	ReKi	Q_p	-	-	-	"Current pump flow rate"
typedef	^	ContinuousStateType	ReKi	Q_m	-	-	-	"Current moter flow rate"
typedef	^	ContinuousStateType	R8Ki	HL_state	21	-	-	"Current estimation of the state vector of the pump line"
```

~~This file will be used to automatically create these variables in the type named `ED_ContinuousStateType` in the file named `ElastoDyn.f90`~~

These variables will be created in the file name `ElastoDyn_Types.f90`, locating at `ED_ContinuousStateType`(533-537):

```fortran
REAL(ReKi)  :: P_p      !< Current pressure difference across the pump [-]
REAL(ReKi)  :: P_m      !< Current pressure difference across the moter [-]
REAL(ReKi)  :: Q_p      !< Current pump flow rate [-]
REAL(ReKi)  :: Q_m      !< Current moter flow rate [-]
REAL(R8Ki) , DIMENSION(1:21)  :: HL_state      !< Current estimation of the state vector of the pump line [-]
```

## Initialize state variable

The state variables should be initialized in the subroutine named `Init_ContStates`(3816-3820)  in the file named `ElastoDyn.f90`. The initialization code is:

```fortran
x%Q_p = 0.04
x%Q_m = 0.05
x%P_p = 1.0E6
x%P_m = 1.0E5
x%HL_state = [real :: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
```

## Calculate $P_p, P_m$ and $Q_p, Q_m$ by the transmission equation

2. ~~Define the parameters required by the transmission equation in the file name`ElastoDyn_Registry.txt`(803-811), locating at~~:

   ```fortran
   # ..... Parameters ................................................................................................................
   # Define parameters here:
   ```
   
   ~~The code is:~~
   
   ```fortran
   typedef	^	ParameterType	ReKi	Omega_m	-	376.8	-	"Motor speed"
   typedef	^	ParameterType	ReKi	D_p	-	626	-	"Pump displacement"
   typedef	^	ParameterType	ReKi	D_m	-	4.9	-	"Moter displacement"
   typedef	^	ParameterType	ReKi	B_p	-	5.0E4	-	"Viscous damping of pump"
   typedef	^	ParameterType	ReKi	B_m	-	2.5	-	"Viscous damping of motor"
   typedef	^	ParameterType	ReKi	C_sp	-	7.1E-11	-	"Laminar leakage coefficients of the pump"
   typedef	^	ParameterType	ReKi	C_sm	-	7.0E-11	-	"Laminar leakage coefficients of the motor"
   typedef	^	ParameterType	ReKi	C_fp	-	0.02	-	"Coulomb friction coefficient of the pump"
   typedef	^	ParameterType	ReKi	C_fm	-	0.02	-	"Coulomb friction coefficient of the motor"
   ```
   
   Define the parameters required by the transmission equation in the file name`ElastoDyn_Types.f90`, locating at `TYPE, PUBLIC :: ED_ParameterType`(806-814):
   
   ```fortran
   REAL(ReKi)  :: Omega_m = 376.8      !< Motor speed [-]
   REAL(ReKi)  :: D_p = 626      !< Pump displacement [-]
   REAL(ReKi)  :: D_m = 4.9      !< Moter displacement [-]
   REAL(ReKi)  :: B_p = 5.0E4      !< Viscous damping of pump [-]
   REAL(ReKi)  :: B_m = 2.5      !< Viscous damping of motor [-]
   REAL(ReKi)  :: C_sp = 7.1E-11      !< Laminar leakage coefficients of the pump [-]
   REAL(ReKi)  :: C_sm = 7.0E-11      !< Laminar leakage coefficients of the motor [-]
   REAL(ReKi)  :: C_fp = 0.02      !< Coulomb friction coefficient of the pump [-]
   REAL(ReKi)  :: C_fm = 0.02      !< Coulomb friction coefficient of the motor [-]
   ```
   
3. Define the subroutine of transmission equation (8569-8622)

   ```fortran
   SUBROUTINE TransDyn(p, x)
      ! This subroutine is used to compute the state of hydraulic line
      IMPLICIT NONE
        TYPE(ED_ParameterType),       INTENT(IN   )  :: p           !< Parameters
        TYPE(ED_ContinuousStateType), INTENT(INOUT)  :: x           !< Continuous states at Time
        REAL, DIMENSION(21,21) :: Phi
        REAL, DIMENSION(441) :: Phi_f
        REAL, DIMENSION(21,2) :: Fa
        REAL, DIMENSION(2,21) :: C
        REAL, DIMENSION(2) :: Q_Mat
        REAL, DIMENSION(2) :: P_Mat
        
        Phi_f = [real :: 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.11908, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99743,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,0.014072,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063, &
        0.99637,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.0016606,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99556,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,0.00019584,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99487,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2.3087e-05,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, & 
        1,1,1,1.0063,0.99427,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2.7207e-06,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99373,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,3.2055e-07,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063, &
        0.99323,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3.7761e-08,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99277,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,4.4476e-09,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99233,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5.2378e-10,1,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1.0063,0.99192]
   
        Phi = RESHAPE(Phi_f, (/21, 21/))
        Fa = RESHAPE((/215754209063.19, -20007901222.68, 211941781201.96, &
        -23820329083.91,214581993601.74, -21180116684.12, 208111716431.86, &
        -12365408591.35, 212562196113.00, -16815888272.49, 204279340001.90, &
        -31482770283.96, 210379894342.47, -25382215943.40, 200445766968.74, & 
        -4699459128.24, 207804157269.94, -12057849429.43,196611429240.20, & 
        -39150681045.66, 205392131396.82, -30369978889.04, 192776548324.89, &
        2969759515.62, 202536000147.97, -6789692307.46, 188941255809.10, &
        -46820854476.77, 200040426182.73, -35721684103.14, 185105637428.56, &
        10640670411.95, 196993048942.12, -1246741101.61, 181269752735.53, &
        -54492357550.33, 194472993417.37, -41289116868.49, 177433645089.65, &
        18312662750.86, 191275713067.87, 4470594772.69/), (/21, 2/))
        C = RESHAPE((/1, 0, 1, 0, 1, 0,	1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, &
        1, 0, 1, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, &
         0, 1/), (/2, 21/))
        Q_Mat = (/x%Q_p, x%Q_m/)
        
        P_Mat = MATMUL(C, x%HL_state)
        x%P_p = P_Mat(1)
        x%P_m = P_Mat(2)
        
        x%HL_state = MATMUL(Phi, x%HL_state) + MATMUL(Fa,Q_Mat)
        x%Q_p = p%D_p * x%QDT(DOF_GeAz) - p%C_sp * x%P_p
        x%Q_m = p%D_m * p%Omega_m + p%C_sm * x%P_m
   
   END SUBROUTINE TransDyn
   ```

3. Call the function in the subroutine named `ED_UpdateStates`(441):

   ```fortran
   CALL TransDyn(p, x)
   ```

4. Calculate the torque of pump and the power produced by the turbine  in the subroutine named `FillAugMat`(8171):

   ```fortran
   GBoxTrq  = p%D_p * x%P_p + p%B_p * x%QDT(DOF_GeAz) + p%C_fp * p%D_p * x%P_p
   ```

5. Calculate the power produced by the turbine  in the subroutine named `ED_CalcOutput`(1100):

   ```fortran
   m%AllOuts( HSShftTq) = p%D_m * x%P_m - p%B_m * p%Omega_m + p%C_fm * p%D_m * x%P_m
   ```


# First revision

In the first version, I set the value of $D_m$ as a constant. However, actuallly, we should control the torque of generator by altering $D_m$. And the module named `elastodyn` only accepts the $Torque$ of generator, rather than $D_m$. Therefore, in this revision, I plan to replace the dynamics of gearbox based drivetrain with the hydraulic drivetrain in the module named `servodyn`, insteading of `elastodyn`. Then, `servodyn` will send the torque of pump to `elastodyn`. It means we do not need to do any modifications in `elastodyn`.

Author: Yubo Huang

Date: 13 July 2021

## Pre-modification

1. Change to the file of `/Users/huangyubo/openfast/reg_tests/r-test/glue-codes/fast-farm/TSinflow/NRELOffshrBsline5MW_Onshore_ElastoDyn_8mps.dat`

2. close the drivetrain DOF 

   ```fortran
   False	      DrTrDOF     - Drivetrain rotational-flexibility DOF (flag)
   ```

   

3. Regard the generator DOF as the Pump DOF and set $n=1$ (Gearbox ratio)

   ```fortran
   3680   GenIner     - Generator inertia about HSS (kg m^2)
   1   GBRatio     - Gearbox ratio (-)
   ```


4. May be the `spd_trq.dat` file `/reg_tests/r-test/glue-code/openfast/.../spd_trq.dat`should also be modified according to the type of turbine used. 

## Define the state variables of the hydralic line 

The state variables of the hydralic line used in the calculation are: 

$$P_p, P_m, Q_p, Q_m, HL_{state}$$

Where $HL_{state}$ is the state vector $x$ in:

$$P = C \times x$$

These variables will be created in the file name `ServoDyn_Types.f90`, locating at `SrvD_ContinuousStateType`(240-251):

```fortran
REAL(ReKi)  :: P_p      !< Current pressure difference across the pump [-]
REAL(ReKi)  :: P_m      !< Current pressure difference across the moter [-]
REAL(ReKi)  :: Q_p      !< Current pump flow rate [-]
REAL(ReKi)  :: Q_m      !< Current moter flow rate [-]
REAL(ReKi)  :: D_m      !< Moter displacement [-]
REAL(R8Ki) , DIMENSION(21, 1)  :: HL_state      !< Current estimation of the state vector of the pump line [-]
```

## Initialize state variable

The state variables should be initialized in the subroutine named `SrvD_Inits`(407-409)  in the file named `ServoDyn.f90`. The initialization code is:

```fortran
x%Q_p = 0.04
x%Q_m = 0.05
x%P_p = 1.0E6
x%P_m = 1.0E5
x%D_m = 0.000986
x%HL_state = RESHAPE((/1.87e7, -1.09e9, 716.05, 2.74e-08, 3.79e-15, -9.78e8, 213.24, 1.58e-08, -1.43e-16, -768.97, 100.52, & 
1.43e-08, 1.80e-15, -5.04e8, 47.12, 4.14e-09, 7.69e-16, -2.33e8, 16.94, 1.15e-09, 2.7e-16/), (/21, 1/))
```

## Calculate $P_p, P_m$ and $Q_p, Q_m$ by the transmission equation

1. Define the parameters required by the transmission equation in the file name`ServoDyn_Types.f90`, locating at `TYPE, PUBLIC :: SrvD_ParameterType`(300):

   ```fortran
   CHARACTER(1024)  :: FSTPath       !< Path Name for the .fst primary input file [-]
   REAL(ReKi)  :: Omega_m = 376.8      !< Motor speed [-]
   REAL(ReKi)  :: D_p = 0.0996      !< Pump displacement [-]
   REAL(ReKi)  :: B_p = 5.0E4      !< Viscous damping of pump [-]
   REAL(ReKi)  :: B_m = 2.5      !< Viscous damping of motor [-]
   REAL(ReKi)  :: C_sp = 7.1E-11      !< Laminar leakage coefficients of the pump [-]
   REAL(ReKi)  :: C_sm = 7.0E-11      !< Laminar leakage coefficients of the motor [-]
   REAL(ReKi)  :: C_fp = 0.02      !< Coulomb friction coefficient of the pump [-]
   REAL(ReKi)  :: C_fm = 0.02      !< Coulomb friction coefficient of the motor [-]
   ```
   
   Initial `p%FSTPath` in the subroutine named`SrvD_Init` in `ServoDyn.f90`, Line 170:
   
   ```fortran
   p%FSTPath  = PriPath 
   ```
   
2. Define the subroutine of transmission equation in `ServoDyn.f90`:

   ```fortran
   SUBROUTINE TransDyn(p, Omega_r, x)
      ! This subroutine is used to compute the state of hydraulic line
      IMPLICIT NONE
        TYPE(SrvD_ParameterType),       INTENT(IN   )  :: p           !< Parameters
        REAL(ReKi), 							 INTENT(IN   )  :: Omega_r 		 !< rotor speed, rad/s.
        ! REAL(ReKi), 							 INTENT(IN   )  :: D_m    		 !< Current displacement of motor.
        TYPE(SrvD_ContinuousStateType), INTENT(INOUT)  :: x           !< Continuous states at Time
        REAL(ReKi), DIMENSION(21,21) :: Phi
        REAL(ReKi), DIMENSION(441) :: Phi_f
        REAL(ReKi), DIMENSION(21,2) :: Fa
        REAL(ReKi), DIMENSION(2,21) :: C
        REAL(ReKi), DIMENSION(2,1) :: Q_Mat
        REAL(ReKi), DIMENSION(2,1) :: P_Mat
        
        Phi_f = [real :: 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.11908, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99743,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,0.014072,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063, &
        0.99637,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.0016606,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99556,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,0.00019584,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99487,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2.3087e-05,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, & 
        1,1,1,1.0063,0.99427,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2.7207e-06,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99373,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,3.2055e-07,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063, &
        0.99323,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3.7761e-08,1,1,1,1,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99277,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,4.4476e-09,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1.0063,0.99233,1,1,1, &
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,5.2378e-10,1,1,1,1,1,1,1,1,1,1,1,1,1,1, &
        1,1,1,1,1,1.0063,0.99192]
   
        Phi = RESHAPE(Phi_f, (/21, 21/))
        Fa = RESHAPE((/215754209063.19, -20007901222.68, 211941781201.96, &
        -23820329083.91,214581993601.74, -21180116684.12, 208111716431.86, &
        -12365408591.35, 212562196113.00, -16815888272.49, 204279340001.90, &
        -31482770283.96, 210379894342.47, -25382215943.40, 200445766968.74, & 
        -4699459128.24, 207804157269.94, -12057849429.43,196611429240.20, & 
        -39150681045.66, 205392131396.82, -30369978889.04, 192776548324.89, &
        2969759515.62, 202536000147.97, -6789692307.46, 188941255809.10, &
        -46820854476.77, 200040426182.73, -35721684103.14, 185105637428.56, &
        10640670411.95, 196993048942.12, -1246741101.61, 181269752735.53, &
        -54492357550.33, 194472993417.37, -41289116868.49, 177433645089.65, &
        18312662750.86, 191275713067.87, 4470594772.69/), (/21, 2/))
        C = RESHAPE((/1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, &
        1, 0, 1, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, &
         0, 1/), (/2, 21/))
        Q_Mat = RESHAPE((/x%Q_p, x%Q_m/), (/2, 1/))
        
        P_Mat = MATMUL(C, x%HL_state)
        x%P_p = P_Mat(1, 1)
        x%P_m = P_Mat(2, 1)
        
        x%HL_state = MATMUL(Phi, x%HL_state) + MATMUL(Fa, Q_Mat)
        x%Q_p = p%D_p * Omega_r - p%C_sp * x%P_p
        x%Q_m = x%D_m * p%Omega_m + p%C_sm * x%P_m
   
   END SUBROUTINE TransDyn
   ```

3. Call the function in the subroutine named `Torque_UpdateStates` in `ServoDyn.f90`(3153):

   ```fortran
   CALL TransDyn(p, u%RotSpeed, D_m, x)
   ```

4. Calculate the torque of pump accourding to the current state `x`. Here `GenTrq` in `CalculateTorque` represents `PumpTrq`. And thus the `GboxTrq` which corresponds to `PumpTrq` should be (8163 in `ElastoDyn.f90`):

```fortran
GBoxTrq    = u%GenTrq
```

(1)	Redefine the the subroutine named `CalculateTorque`(3160)  in `ServoDyn.f90` (deliver `x` to this subroutine):

```fortran
SUBROUTINE CalculateTorque( t, u, p, x, D_m, m, GenTrq, ElecPwr, ErrStat, ErrMsg )
	TYPE(SrvD_ContinuousStateType),  INTENT(IN   ) :: x           !< Input: Continuous states at t;
	REAL(ReKi)  										 INTENT(OUT  ) :: D_m    		  !< Current displacement of motor.
	D_m     = 0.0_ReKi
```

(2)	Thus, the places where call `CalculateTorque` should all be modified:

**Line 970 and 1117**  

```fortran
REAL(ReKi)  										 								 :: D_m    		  !< Current displacement of motor.
CALL CalculateTorque( t, u_interp, p, x, D_m, m, m%dll_data%GenTrq_prev, m%dll_data%ElecPwr_prev, ErrStat2, ErrMsg2 )
```

**Line 2982 and 2996**

   ```fortran
REAL(ReKi)  										 								 :: D_m    		  !< Current displacement of motor.
   CALL CalculateTorque( t, u, p, x, D_m, m, y%GenTrq, y%ElecPwr, ErrStat, ErrMsg )
   ```

   **Line 3105 and 3147**

```fortran
   REAL(ReKi)  										 								 :: D_m    		  !< Current displacement of motor.
CALL CalculateTorque( t, u, p, x, D_m, m, GenTrq, ElecPwr, ErrStat, ErrMsg ) !< This function should be changed to Line 3153
x%D_m = D_m
```

   (3)	Deliver `x%P_p,` `x%P_m,` `u%RotSpeed` to subroutines named `UserGen` called by `CalculateTorque`(3251) in`ServoDyn.f90`:

   ```fortran
CALL UserGen ( u%RotSpeed, x%P_p, x%P_m, p%NumBl, t, p%DT, p%GenEff, 0.0_ReKi, p%RootName, GenTrq, ElecPwr, D_m )
   ```

   (4)	Deliver `x%P_p,` `x%P_m,` `u%RotSpeed` to subroutines named `UserVSCont` called by `CalculateTorque`(3286) in `ServoDyn.f90`:

   ```fortran
   CALL UserVSCont ( u%RotSpeed, x%P_p, x%P_m, p%NumBl, t, p%DT, p%GenEff, 0.0_ReKi, p%RootName, GenTrq, ElecPwr, D_m )
   ```

   Now, the modifications in `servodyn.f90` and ``ServoDyn_Types.f90``have been completed and then I will modify the corresponding `UserGen` and `UserVSCont` in the file named `UserVSont_KP.f90` to obtain `GenTrq(PumpTrq)` and `ElecPwr`

## Calculate `GenTrq(PumpTrq)` for `ElastoDyn` and `ElecPwr` for output

### Modify the file named `UserVSCont_KP.f90`

(1) Modify the subroutine named `UserGen`:

Line 13:

From:

```fortran
SUBROUTINE UserGen ( HSS_Spd, LSS_Spd, NumBl, ZTime, DT, GenEff, DelGenTrq, DirRoot, GenTrq, ElecPwr )
```

To

```fortran
SUBROUTINE UserGen ( Rot_Spd, P_pump, P_motor, NumBl, ZTime, DT, GenEff, DelGenTrq, DirRoot, GenTrq, ElecPwr, D_m )
```

Delete Line 36 - 37:

```fortran
REAL(ReKi), INTENT(IN )     :: LSS_Spd                                       ! LSS speed, rad/s.
REAL(ReKi), INTENT(IN )     :: HSS_Spd                                       ! HSS speed, rad/s.
```

Line 36 - 28, add:

```fortran
REAL(ReKi), INTENT(IN )     :: Rot_Spd                                       ! Rotor speed, rad/s.
REAL(ReKi), INTENT(IN )     :: P_pump                                        ! Pressure difference in pump lines, pa.
REAL(ReKi), INTENT(IN )     :: P_motor                                       ! Pressure difference in motor lines, pa.
```

Line 40, add:

```fortran
REAL(ReKi), INTENT(OUT)     :: D_m                                           ! Current Motor displacement
```

Delete, Line 46:

```fortran
CALL UserVSCont ( Rot_Spd, P_pump, P_motor, NumBl, ZTime, DT, GenEff, DelGenTrq, DirRoot, GenTrq, ElecPwr, D_m )   ! Let's have UserGen() do the same thing as SUBROUTINE UserVSCont().
```

Line 46, add:

```fortran
CALL UserVSCont ( Rot_Spd, P_pump, P_motor, NumBl, ZTime, DT, GenEff, DelGenTrq, DirRoot, GenTrq, ElecPwr, D_m )   ! Let's have UserGen() do the same thing as SUBROUTINE UserVSCont().
```

Delete, Line 53:

```fortran
SUBROUTINE UserVSCont ( HSS_Spd, LSS_Spd, NumBl, ZTime, DT, GenEff, DelGenTrq, DirRoot, GenTrq, ElecPwr )
```

Lind 53, add:

```fortran
SUBROUTINE UserVSCont ( Rotor_Spd, P_pump, P_motor, NumBl, ZTime, DT, GenEff, DelGenTrq, DirRoot, GenTrq, ElecPwr, D_m )
```

Delete, Line 81:

```fortran
REAL(ReKi), INTENT(IN )     :: LSS_Spd                                       ! LSS speed, rad/s.
```

Line 81, add:

```fortran
REAL(ReKi), INTENT(IN )     :: Rotor_Spd                                       ! Rotor speed, rad/s.
```

Delete, Line 84:

```fortran
REAL(ReKi), INTENT(IN )     :: HSS_Spd                                       ! HSS speed, rad/s.
```

Line 84-85, add:

```fortran
REAL(ReKi), INTENT(IN )     :: P_pump                                        ! Pressure difference in pump lines, pa.
REAL(ReKi), INTENT(IN )     :: P_motor                                       ! Pressure difference in motor lines, pa.
```

Line 87, add:

```fortran
REAL(DbKi), INTENT(OUT)     :: D_m                                           ! Current Motor displacement
```

Delete, Line 97-98:

```fortran
REAL(ReKi), SAVE            :: FTRQ    = 0.0                                 ! Filtered torque, N-m.
REAL(ReKi), SAVE            :: OLTRQ   = 0.0
```

Delete, Line 99:

```fortran
REAL(ReKi), SAVE            :: RPMSCH  (100)
```

Delete, Line 102-104:

```fortran
REAL(ReKi), SAVE            :: TRQ     = 0.0
REAL(ReKi), SAVE            :: TRQSCH  (100)
REAL(DbKi), SAVE            :: TTRQ    = 0.0
```

Delete, Line 105-106:

```fortran
INTEGER(IntKi)              :: N1
INTEGER, SAVE               :: NSCH   = 0                                    ! Number of lines found in the file
```

Line 116-147, add:

```fortran
REAL(ReKi)                  :: D_p = 626                                     ! Pump displacement
REAL(ReKi)                  :: B_p = 50000                                   ! Viscous damping of pump
REAL(ReKi)                  :: B_m = 2.5                                     ! Viscous damping of motor
REAL(ReKi)                  :: C_fp = 0.02                                   ! Viscous damping of motor
REAL(ReKi)                  :: C_fm = 0.02                                   ! Viscous damping of motor
REAL(ReKi)                  :: Omega_m = 376.8                               ! Constant speed of motor, rad/s, 2*pi*f
REAL(ReKi)                  :: Trq_Dem                                       ! the demanded motor torque
REAL(ReKi)                  :: D_m_Dem                                       ! Viscous damping of motor
REAL(ReKi), SAVE            :: MotTrq                                        ! Viscous damping of motor
REAL(ReKi), SAVE            :: D_m_ZT = 0.0                                  ! Motor displacement at Ztime

! The parameter of MLP
! There are four layers in the MLP, and the activation function is tanh
! Input layer: there are four states: ID of the turbine, rotor speed, P_pump, P_motor
! First hidden layer: there are n1 neural units in this layer
! Second hidden layer: there are n1 neural units in this layer
! Output layer: the output is the reference of D (Displacement)

! ! The location in wind farm of this turbine
! REAL(ReKi), SAVE :: Loc_x
! REAL(ReKi), SAVE :: Loc_y
! REAL(ReKi), SAVE :: Loc_z   
INTEGER(IntKi), SAVE :: n1    ! n1 denotes the number of neural units in the first hidden layer
INTEGER(IntKi), SAVE :: n2    ! n2 denotes the number of neural units in the second hidden layer
REAL(ReKi)    :: x(1)  ! the input vector of MLP, containing four states of turbine
REAL(ReKi), SAVE, DIMENSION(:,:), ALLOCATABLE :: w1     ! The weight matrix between the input layer and the first hidden layer 
REAL(ReKi), SAVE, DIMENSION(:,:), ALLOCATABLE :: b1     ! The bias vector between the input layer and the first hidden layer
REAL(ReKi), SAVE, DIMENSION(:,:), ALLOCATABLE :: w2     ! The number of neural units in the second hidden layer
REAL(ReKi), SAVE, DIMENSION(:,:), ALLOCATABLE :: b2     ! The bias vector between the first hidden layer and the second hidden layer
REAL(ReKi), SAVE, DIMENSION(:,:), ALLOCATABLE :: w3     ! The weight matrix between the second hidden layer and the output layer
REAL(ReKi), SAVE, DIMENSION(:,:), ALLOCATABLE :: b3     ! The bias vector between the second hidden layer and the output layer

```

Delete, Line 152-155:

```fortran
IF ( .NOT. EqualRealNos( HSS_Spd, LSS_Spd ) )  THEN
   CALL ProgAbort ( " GBRatio must be set to 1.0 when using Kirk Pierce's UserVSCont() routine." )
END IF
OMEGA = HSS_Spd
```

Line 154-160, add:

```fortran
OMEGA = Rotor_Spd
! Compute the pump torque
! Alghough the name is GenTrq, it represents the pump torque
GenTrq = D_p * P_pump + B_p * OMEGA + C_fp * D_p * P_pump 
GenTrq = MIN(MAX(GenTrq, 10.0), 20000.0) 
    
! Compute the demanded motor torque according to the current rotor speed
```

Delete, Line 179-209:

```fortran
DO I=1,100
      READ(UnCont,*,IOSTAT=IOS)  RPMSCH(I), TRQSCH(I)
      IF ( IOS < 0 )  EXIT

      IF ( I > 1 ) THEN
         IF ( RPMSCH(I) <= RPMSCH(I-1) ) THEN
            CALL ProgWarn('RPM schedule must be increasing in file spd_trq.dat. Schedule will be stopped at ' &
                                   //TRIM(Num2LStr(RPMSCH(I-1)))//' RPM.')
            EXIT
         END IF
      END IF
      NSCH = NSCH + 1
END DO ! I

SMPDT = REAL( NST, DbKi )*DT

   C1 = EXP( -DT/TCONST )
   C2 = 1.0 - C1

   SFLAG = .FALSE.
   CLOSE(UnCont)

   IF ( NSCH < 2 ) THEN
      IF ( NSCH == 0 ) THEN
         RPMSCH(1) = 0.0
         TRQSCH(1) = 0.0
      END IF
      NSCH = 2
      RPMSCH(2) = RPMSCH(1)
      TRQSCH(2) = TRQSCH(1)
   END IF

```

Line 181 - 224, add:

```fortran
   ! READ (UnCont, *) Loc_x, Loc_y, Loc_z

   READ (UnCont, *) n1, n2
   ALLOCATE(w1(n1, 1))
   ALLOCATE(b1(n1, 1))
   ALLOCATE(w2(n2, n1))
   ALLOCATE(b2(n2, 1))
   ALLOCATE(w3(2, n2))
   ALLOCATE(b3(2, 1))

   ! Read w1
   DO I = 1, n1
      READ(UnCont, *) w1(I, :)
   END DO

   ! Read b1
   DO I = 1, n1
      READ(UnCont, *) b1(I, 1)
   END DO

   ! Read w2
   DO I = 1, n2
      READ(UnCont, *) w2(I, :)
   END DO

   ! Read b2
   DO I = 1, n2
      READ(UnCont, *) b2(I, 1)
   END DO

   ! Read w3
   READ(UnCont, *) w3(1, :)
   READ(UnCont, *) w3(2, :)

   ! Read b3
   READ(UnCont, *) b3(1, 1)
   READ(UnCont, *, IOSTAT=ios) b3(2, 1)
    
   SFLAG = .FALSE.
   CLOSE(UnCont)
   
ENDIF

! The location information of turbine
! x(1) = Loc_z
! x(2) = Loc_y
! x(3) = Loc_z
```

Delete, Line 224-248:

```fortran
   ! Calculate torque setting at every NST time steps.
IF ( EqualRealNos( DELT, ( SMPDT - 0.5_DbKi*DT ) ) .OR. (DELT > ( SMPDT - 0.5*DT ))) then
!IF ( DELT >= ( SMPDT - 0.5*DT ) )  THEN !this should be comparing with EqualRealNos()

   TLST = ZTime  !BJJ: TLST is a saved variable, which may have issues on re-initialization.


   ! Update old values.

   DO I=5,2,-1
      FRPM(I) = FRPM(I-1)
   ENDDO ! I

   RPM = OMEGA * 30.0/PI

   ! Calculate recursive lowpass filtered value.

   FRPM(1) = 0.7*FRPM(2) + 0.3*RPM
   FRPM(1) = MIN( MAX( FRPM(1), RPMSCH(1) ), RPMSCH(NSCH) )
   TRQ     = InterpBin( FRPM(1), RPMSCH(1:NSCH), TRQSCH(1:NSCH), N1, NSCH )


ENDIF
```

Lind 226-267, add:

```fortran
DELT = ZTime - TLST
IF ( EqualRealNos( DELT, 0.5_DbKi*DT ) .OR. (DELT > ( 0.5*DT ))) then
!IF ( DELT >=  0.5*DT )  THEN !this should be comparing with EqualRealNos()

   TLST = ZTime  !BJJ: TLST is a saved variable, which may have issues on re-initialization.


   ! Update old values.
   DO I=5,2,-1
      FRPM(I) = FRPM(I-1)
   ENDDO ! I

   RPM = OMEGA * 30.0/3.14
    
   !  Calculate recursive lowpass filtered value.
    
   FRPM(1) = 0.7*FRPM(2) + 0.3*RPM
   FRPM(1) = MIN( MAX( FRPM(1), 0.0 ), 1000.0 )
   x(1) = FRPM(1)
   ! compute the reference of motor torque
   CALL MLP(x, n1, n2, w1, b1, w2, b2, w3, b3, Trq_Dem)
   ! compute the reference of motor displacement
   D_m_Dem = (Trq_Dem + B_m * Omega_m) / ((1 + C_fm) * P_motor)
   ! motor displacement actuator 
   CALL PID(D_m_Dem, DT, D_m_ZT)

ENDIF

D_m = D_m_ZT
   
MotTrq = D_m * P_motor - B_m * Omega_m + C_fm * D_m * P_motor

! GenTrq = FTRQ + DelGenTrq  ! Make sure to add the pertubation on generator torque, DelGenTrq.  This is used only for FAST linearization (it is zero otherwise).
    
! The generator efficiency is either additive for motoring,
!   or subtractive for generating power.
    
IF ( MotTrq > 0.0 )  THEN
   ElecPwr = MotTrq*Omega_m*GenEff
ELSE
   ElecPwr = MotTrq*Omega_m/GenEff
ENDIF
```

Delete 254-274:

```fortran
   ! Torque is updated at every integrator time step
IF ( (.NOT. EqualRealNos(ZTime, TTRQ )) .AND. ZTime > TTRQ )  THEN
!IF ( ZTime > TTRQ )  THEN

   FTRQ  = C1*FTRQ + C2*OLTRQ
   OLTRQ = TRQ
   TTRQ  = ZTime + 0.5_DbKi*DT

ENDIF


GenTrq = FTRQ + DelGenTrq  ! Make sure to add the pertubation on generator torque, DelGenTrq.  This is used only for FAST linearization (it is zero otherwise).

   ! The generator efficiency is either additive for motoring,
   !   or subtractive for generating power.

IF ( GenTrq > 0.0 )  THEN
   ElecPwr = GenTrq*HSS_Spd*GenEff
ELSE
   ElecPwr = GenTrq*HSS_Spd/GenEff
ENDIF
```

Line 272-354, add:

```fortran
SUBROUTINE PID(D_m_Dem, DT, D_m)
   USE                            NWTC_Library
    IMPLICIT NONE 
    REAL(ReKi), INTENT(IN )       :: D_m_Dem
    REAL(DbKi), INTENT(IN )       :: DT
    REAL(ReKi), INTENT(OUT)       :: D_m 
    REAL(ReKi), PARAMETER         :: Kp = -0.85
    REAL(ReKi), PARAMETER         :: Ki = 0.036
    REAL(ReKi), PARAMETER         :: Kd = 0.0
    REAL(ReKi), SAVE              :: U1 = 0.0
    REAL(ReKi), SAVE              :: Y1 = 0.0
    REAL(ReKi), SAVE              :: X(3) = (/0.0, 0.0, 0.0/)
    REAL(ReKi)                    :: ERROR
    REAL(ReKi), SAVE              :: ERROR1 = 0.0
    REAL(ReKi)                    :: U 
   
   ! Compute the output of PID controller
    U = Kp * X(1) + Kd * X(2) + Ki * X(3)

   ! Limit the output scope of PID controller
    IF (U > 10.0) THEN
        U = 10.0
    END IF 

    IF (U < -10.0) THEN
        U = -10.0
    END IF 

   ! The acturator
    D_m = 0.9394 * Y1 + 0.06259 * U1
   ! Limit the scope of D_m
    D_m = MIN(MAX(D_m, 0.0000001), 0.1)
   ! compute the error between the current error and its reference 
    ERROR = D_m_Dem - D_m
   ! record outputs of thePID controller and the entire control system
    U1 = U 
    Y1 = D_m 

   ! The input of P, D and I
    X(1) = ERROR
    X(2) = (ERROR - ERROR1) / DT 
    X(3) = X(3) + ERROR * DT 

    ERROR1 = ERROR

END SUBROUTINE PID
  
SUBROUTINE MLP(x, n1, n2, w1, b1, w2, b2, w3, b3, y)
      USE                            NWTC_Library
      IMPLICIT NONE
      REAL(ReKi), INTENT(IN)     :: x(1)     ! The input of MLP (Current state)
      INTEGER(IntKi), INTENT(IN) :: n1       ! The number of neural units in the first hidden layer
      REAL(ReKi), INTENT(IN)     :: w1(n1, 1) ! The weight matrix between the input layer and the first hidden layer
      REAL(ReKi), INTENT(IN)     :: b1(n1, 1)   ! The bias vector between the input layer and the first hidden layer
      INTEGER(IntKi), INTENT(IN) :: n2       ! The number of neural units in the second hidden layer
      REAL(ReKi), INTENT(IN)     :: w2(n2, n1) ! The weight matrix between the first hidden layer and the second hidden layer
      REAL(ReKi), INTENT(IN)     :: b2(n2, 1) ! The bias vector between the first hidden layer and the second hidden layer
      REAL(ReKi), INTENT(IN)     :: w3(2, n2) ! The weight matrix between the second hidden layer and the output layer
      REAL(ReKi), INTENT(IN)     :: b3(2, 1)  ! The bias vector between the second hidden layer and the output layer
      REAL(ReKi), INTENT(OUT)    :: y        ! The output of MLP
      
      ! Local variables
      REAL(ReKi)    :: x0(1, 1)
      REAL(ReKi)    :: x1(n1, 1)   ! The state vector in the first hidden layer
      REAL(ReKi)    :: x2(n2, 1)   ! The state vector in the second hidden layer
      ! REAL(ReKi)    :: y_tem(2, 1) ! Due to the matrix can not directlt add integer
      REAL(ReKi)    :: x3(2, 1)    ! The state vector in the output layer
      REAL(ReKi)    :: min = 0.0 ! The minimum of output
      REAL(ReKi)    :: max = 15000.0  ! The maximum of output
      REAL(ReKi)    :: rand_std_normal_num
  
      ! The first input layer
      x0 = reshape(x, (/1, 1/))
      ! The first hidden layer
      x1 = matmul(w1, x0) + b1 
      x1 = tanh(x1)
      ! The second hidden layer
      x2 = matmul(w2, x1) + b2
      x2 = tanh(x2)
      ! The output layer
      x3 = matmul(w3, x2) + b3 
      x3 = tanh(x3)

      ! Generate a random number base on the std normal distribution
      CALL RandomSTDNormal(rand_std_normal_num)
      ! Transform the std random number into the random number ruled by distribution x3
      x3(2, 1) = min(max(x3(2, 1), -16.0), 16.0)
      y = rand_std_normal_num * exp(x3(2, 1)) + x3(1, 1) 
  
      ! Limit the output scope
      y = min(max(y, min_num), max_num)
  
END SUBROUTINE MLP

SUBROUTINE RandomSTDUniform(u)
   USE                            NWTC_Library
   IMPLICIT NONE
   REAL(ReKi), INTENT(OUT)    :: u
   REAL(ReKi)                 :: r 

   CALL random_number(r)
   u = 1 - r 
ENDSUBROUTINE RandomSTDUniform

SUBROUTINE RandomSTDNormal(x)
   USE                            NWTC_Library
   IMPLICIT NONE
   REAL(ReKi), INTENT(OUT)    :: x 
   REAL(ReKi), PARAMETER      :: pi = 3.14159265
   REAL(ReKi)                 :: u1, u2

   CALL RandomSTDUniform(u1)
   CALL RandomSTDUniform(u2)

   x = sqrt(-2*log(u1))*cos(2*pi*u2)
ENDSUBROUTINE RandomSTDNormal
```



