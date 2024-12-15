/*********************************************
 * OPL 22.1.1.0 Model
 * Author: gozalid
 * Creation Date: 6 Mar. 2024 at 10:59:18 am
 *********************************************/
 main {
   thisOplModel.generate();
//   thisOplModel.printSolution();
   if (cplex.solve()) {
     var ofile = new IloOplOutputFile("Y:/Sentosa/Optimization_Algos/Het54TripsEVeh1CS_Idle.txt");
     ofile.writeln("Solving CPU Elapsed Time in (Seconds): ", cplex.getCplexTime());
     var obj = cplex.getObjValue();
     ofile.writeln("The value of the objective function is (Total Cost): ", obj);
     ofile.writeln("");
     ofile.writeln(thisOplModel.Y);
     ofile.writeln("");
     ofile.writeln(thisOplModel.U);
     var K = 0;
     for (var i = 1; i <= thisOplModel.NBuses; i++) {
       if (thisOplModel.U[i] == 1) {
         writeln(i);
         K+=1;
       }
     }
     ofile.writeln("The total number of e-buses dispatched: ", K);
     for (var k = 1; k <= K; k++) {
       ofile.write("BusID " + k + ": " + 0);
       var i = 0, j = -1; c = 0;
       while(j != 0) {
         c+=1;
         for (j = 0; j <= (thisOplModel.NTrips + thisOplModel.NCS); j++) {
           if (c > 3) {
             c = 1;
           }
           if (thisOplModel.Y[i][j][k][c] == 1) {
             ofile.write(" -> " + j + " @ " + c);
             i = j;
             break;
           }
         }
       }
       ofile.writeln("");
     }
     ofile.close();
   } else {
     writeln("No Solution");
   }
 }
  // Parameters
 int NTrips = 54;
 int NBuses = 50;
 int NRCharged = 3; // number of recharging cycle
 int NCS = 1; // Charging Points
 range RBuses = 1..NBuses;
 range RTrips = 0..NTrips; // 0 represents depot
 range RCharged = 1..NRCharged; // Recharging Cycle
 range RCSPoints  = NTrips+1..NTrips+NCS; // Recharging Points
 int MAX_DURATION = ...;
 int TMAX = ...;
 
 // int CHARGING_TIME = ...;
 
 {int} K = asSet(RBuses); // All Buses
 {int} S = asSet(RTrips); //All available schedules where depot is 0
 {int} S1 = S diff{0}; // All available stations
 {int} C = asSet(RCharged); // All available recharge cycles
 {int} R = asSet(RCSPoints);
// {int} R = {NTrips+NCS}; // All available recharge points [LOCATION]
 {int} Sprime = S union R;
 
  // Multiple Recharging Stations//
  range reindex = 0..NCS*(NTrips); // (depot)
//  range reindex = 0..NCS*(NTrips)+1;
// range reindex = 0..NCS*(NTrips)+2; // +2 (depots)
 {int} reCycle = asSet(reindex);
 
 // Idle Time //
 int zeta[S, S] = ...;
// range index_idle = 0..(NCS)*(NTrips)+2; //+2 (1x pair of schedules + 2x reroute to CS)
// {int} idleCycle = asSet(reindex); + 
 
 // Feasible Matrix
 int gamma[S,S] = ...;
 // Duration Matrix between A schedule to the next possible schedule.
 int delta[Sprime,Sprime] = ...; // (delta union cost to cs) the cost between schedules and cost to charging station from the given schedule.
 int actual_dur[Sprime] = ...;
 int MultiCS[reCycle,S] = ...;
// int PHI[S,S] = ...;
 int phi[i in S, j in S, r in R] = MultiCS[(r-NTrips-1) * (NTrips+1)+i, j]; 
 
// //Idle Time Matrix
 int idles[reCycle, S] = ...;
// int zeta[i in Sprime, j in Sprime] = 
 int zeta_charged[i in S, j in S, r in R] = idles[(r-NTrips-1) * (NTrips+1)+i, j];

 
 // Decision Variables
 // binary variable indicates that Y == 1 if Schedule i precedes Schedule j during charge cycle c for bus k with 
 // i, j \in S'
 dvar boolean Y[Sprime,Sprime,K,C];
 // The total number of Buses
 dvar boolean U[K];
 float alpha = 1/(NBuses * TMAX);
 
 minimize sum(k in K) U[k] 
 	+ sum(i in S, j in S diff{i}, r in R) (phi[i,j,r] * (alpha * zeta_charged[i,j,r]) + (1-phi[i,j,r])*(alpha * zeta[i,j]));//zeta[i,j];//zeta[i, j, r-1];
 
 subject to {
   // Cannot exceed the maximum number of visits to recharge station
   // (1) the out-flow must be one (supply constraint)
   forall(j in S diff {0}) sum(i in Sprime diff{j}, k in K, c in C) Y[i,j,k,c] == 1;
   // (2) the in-flow must be one (demand constraint)
   forall(i in S diff {0}) sum(j in Sprime diff{i}, k in K, c in C) Y[i,j,k,c] == 1;
   // (3) the number of in-flow and out-flow must be the same
   forall(j in S diff {0}, k in K, c in C) sum(i in Sprime diff{j}) Y[i,j,k,c] == sum(i in Sprime diff{j}) Y[j,i,k,c];
   //NB: I needed to remove the depot from S
   
   //new
   forall(i in Sprime, j in Sprime diff {i}, k in K, c in C) Y[i,j,k,c] <= U[k];
   
   // (4) Cycle finish (recharge) node be the same as the start of the next cycle. 
   forall(k in K, c in C) sum(i in S, j in R) Y[i,j,k,c] <= 1;
   // (5) Not enough time to go from i to recharge r then to j***
   forall(k in K, r in R, c in C diff {1}) sum(i in S) Y[i,r,k,(c-1)] == sum(j in S) Y[r,j,k,c];
   // (6) Not enough time to do i then j with the same bus
//   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,c-1] + Y[r,j,k,c]) <= (2*PHI[i,r,j]);

	//TODO: reactivate
   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,(c-1)] + Y[r,j,k,c]) <= phi[i, j, r] + 1;
   //NB: 2*PHI was wrong because we are considering all pairs i,j;
   //if we have i->r->j with PHIij=1, all is good, but there will also be a constraint checking other combinations
   //e.g. k->r->j which may have PHIkj=0. This will prevent going to j from r even for other valid combinations
   //since RHS for pair k,j will be <= 0, it will for Yrjkc=0 regardless of anythings
   
   //debugging...
   //Y[0,1,1,1] == 1;
   //Y[1,3,1,1] == 1;
   //Y[3,4,1,1] == 1;
   //Y[4,2,1,2] == 1;
   //Y[2,0,1,2] == 1;
   //Y[3,4,1,1] + Y[4,2,1,2] <= 2;				//this worked
   //Y[3,4,1,1] + Y[4,2,1,2] <= 2 * PHI[3,2];		//so did this
   
//   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,c-1] + Y[r,j,k,c]) <= (2*rPHI[(r-NTrips-1)+i,j]);
   // (7) Feasible constraint
   forall(i in S, j in S diff {i}) sum(k in K, c in C) Y[i,j,k,c] <= gamma[i,j];
//   forall(i in S, j in S diff {i}) sum(k in K, c in {1}) Y[i,j,k,c] <= gamma[i,j];
   // (8) Energy constraint
   forall(k in K, c in C) sum(i in Sprime, j in Sprime diff{i}) delta[i,j] * Y[i,j,k,c] <= MAX_DURATION;
   // (9) Continue Recharging Cycle (NB: REMOVED - THIS WILL CREATE INFEASIBILITY)
   //forall(k in K, i in Sprime, j in Sprime diff{i}, c in C diff{1}) Y[i,j,k,c] <= Y[i,j,k,c-1];
   // (10) Constraint to find only feasible solutions
   //forall(k in K) sum(j in S diff{0}) Y[0,j,k,1] == U[k];
   
   //new: need these to force depot at start and end, other constraint sdidn't cut it'
   forall(k in K, c in C diff{1}, j in Sprime) Y[0,j,k,c] == 0;
   forall(k in K, c in C diff{1}, i in Sprime, j in Sprime diff{i}) Y[i,j,k,c] <= 1 - sum(h in Sprime) Y[h,0,k,c-1];
   forall(k in K, c in C, r in R) Y[0,r,k,c] + Y[r,0,k,c] == 0;
   
   // (11) Continuous Sequence
   forall(k in K diff{1}) U[k] <= U[k-1];
 } 
 
 
 
// main {
//   thisOplModel.generate();
////   thisOplModel.printSolution();
//   if (cplex.solve()) {
//     var ofile = new IloOplOutputFile("Y:/Sentosa/Optimization_Algos/MIP54TripsEVeh3CS_21secs.txt");
////     ofile.writeln(thisOplModel.printSolution());
//     ofile.writeln("Solving CPU Elapsed Time in (Seconds): ", cplex.getCplexTime());
//     var obj = cplex.getObjValue();
//     ofile.writeln("The value of the objective function is (Total number of fleet size): ", obj);
//     ofile.writeln("");
//     ofile.writeln(thisOplModel.Y);
//     ofile.writeln("");
//     ofile.writeln(thisOplModel.U);
//     ofile.close();
//   } else {
//     writeln("No Solution");
//   }
// }
// main {
//   thisOplModel.generate();
////   thisOplModel.printSolution();
////   writeln("Optimality: ", thisOplModel.printSolution());
//   if (cplex.solve()) {
//     writeln("Solving CPU Elapsed Time in (Seconds): ", cplex.getCplexTime());
//     var obj = cplex.getObjValue();
//     writeln("The value of the objective function is (Total number of fleet size): ", obj);
//     for (var k = 1; k <= obj; k++) {
//       write("BusID " + k + ": " + 0);
//       var i = 0, j = -1; c = 1;
//       while(j != 0) {
//         for (j = 0; j <= (thisOplModel.NTrips + thisOplModel.NCS); j++) {
//           if (c > 3) {
//             c = 1;
//           }
//           if (thisOplModel.Y[i][j][k][c] == 1) {
//             write(" -> " + j + " @ " + c);
//             i = j;
//             break;
//           } else {
//             c += 1;
//           }
//         }
//       }
//       writeln("");
//     }
////     var ofile = new IloOplOutputFile("Y:/Sentosa/Optimization_Algos/Hetro30TripsEVeh3CS.txt");
////     ofile.writeln(thisOplModel.printSolution());
////     ofile.writeln("Solving CPU Elapsed Time in (Seconds): ", cplex.getCplexTime());
////     ofile.writeln("The value of the objective function is (Total number of fleet size): ", obj);
////     ofile.writeln("");
////     for (var k = 1; k <= obj; k++) {
////       ofile.write("BusID " + k + ": " + 0);
////       var i = 0, j = -1; c = 1;
////       while(j != 0) {
////         for (j = 0; j <= (thisOplModel.NTrips + thisOplModel.NCS); j++) {
////           if (c > 3) {
////             c = 1;
////           }
////           if (thisOplModel.Y[i][j][k][c] == 1) {
////             ofile.writeln(" -> " + j + " @ " + c);
////             i = j;
////             break;
////           } else {
////             c += 1;
////           }
////         }
////       }
////       ofile.writeln("");
////     }
////     ofile.close();
////     var tolist = new IloOplOutputFile("Y:/Sentosa/Optimization_Algos/result.txt");
////     tolist.write("[")
////     for (var k = 1; k <= obj; k++) {
////       tolist.write("[");
////       var i = 0, j = -1;
////       while(j != 0) {
////         for (j = 0; j <= thisOplModel.NTrips; j++) {
////           if (thisOplModel.Y[i][j][k] == 1) {
////             tolist.write(j);
////             i = j;
////             if (j != 0) {
////               tolist.write(",");
////             }
////             break;
////           }
////         }
////       }
////       tolist.writeln("], ");
////     }
////     tolist.write("]");
////     tolist.close();
//   } else {
//     writeln("No Solution");
//   }
// }
//main {
//   thisOplModel.generate();
////   thisOplModel.printSolution();
//   if (cplex.solve()) {
//     var ofile = new IloOplOutputFile("Y:/Sentosa/Optimization_Algos/Hetro54TripsEVeh3CS_Earliest.txt");
////     ofile.writeln(thisOplModel.printSolution());
//     ofile.writeln("Solving CPU Elapsed Time in (Seconds): ", cplex.getCplexTime());
//     var obj = cplex.getObjValue();
//     ofile.writeln("The value of the objective function is (Total number of fleet size): ", obj);
//     ofile.writeln("");
//     for (var k = 1; k <= obj; k++) {
//       ofile.write("BusID " + k + ": " + 0);
//       var i = 0, j = -1; c = 0;
//       while(j != 0) {
//         c+=1;
//         for (j = 0; j <= (thisOplModel.NTrips + thisOplModel.NCS); j++) {
//           if (c > 3) {
//             c = 1;
//           }
//           if (thisOplModel.Y[i][j][k][c] == 1) {
//             ofile.writeln(" -> " + j + " @ " + c);
//             i = j;
//             break;
//           }
//         }
//       }
//       ofile.writeln("");
//     }
//     ofile.close();
//   } else {
//     writeln("No Solution");
//   }
// }
// // Parameters
// int NTrips = 54;
// int NBuses = 30;
// int NRCharged = 3; // number of recharging cycle
// int NCS = 3; // Charging Points
// range RBuses = 1..NBuses;
// range RTrips = 0..NTrips; // 0 represents depot
// range RCharged = 1..NRCharged; // Recharging Cycle
// range RCSPoints  = NTrips+1..NTrips+NCS; // Recharging Points
// int MAX_DURATION = ...;
// // int CHARGING_TIME = ...;
// 
// {int} K = asSet(RBuses); // All Buses
// {int} S = asSet(RTrips); //All available schedules where depot is 0
// {int} S1 = S diff{0}; // All available stations
// {int} C = asSet(RCharged); // All available recharge cycles
// {int} R = asSet(RCSPoints); // All available recharge points [LOCATION]
// {int} Sprime = S union R;
// 
// // Multiple Recharging Stations//
// range reindex = 0..NRCharged*(NTrips)+2; // +2 (depots)
// {int} reCycle = asSet(reindex);
// 
// // Parameters
// // Feasible Matrix
// int gamma[S,S] = ...;
// // Duration Matrix between A schedule to the next possible schedule.
// int delta[Sprime,Sprime] = ...; // (delta union cost to cs) the cost between schedules and cost to charging station from the given schedule.
// int actual_dur[Sprime] = ...;
//// int PHI[reCycle,S] = ...; // binary variable indicates if recharging is feasible and based on the time and location.
// int MultiCS[reCycle,S] = ...; // binary variable indicates if recharging is feasible and based on the time and location.
// int phi[i in S, j in S, r in R] = MultiCS[(r-NTrips-1) * (NTrips+1)+i, j];  
// 
// // Decision Variables
// // binary variable indicates that Y == 1 if Schedule i precedes Schedule j during charge cycle c for bus k with 
// // i, j \in S'
// dvar boolean Y[Sprime,Sprime,K,C];
// // The total number of Buses
// dvar boolean U[K];
// 
// minimize sum(k in K) U[k];
// 
// subject to {
//      // Cannot exceed the maximum number of visits to recharge station
//   // (1) the out-flow must be one (supply constraint)
//   forall(j in S diff {0}) sum(i in Sprime diff{j}, k in K, c in C) Y[i,j,k,c] == 1;
//   // (2) the in-flow must be one (demand constraint)
//   forall(i in S diff {0}) sum(j in Sprime diff{i}, k in K, c in C) Y[i,j,k,c] == 1;
//   // (3) the number of in-flow and out-flow must be the same
//   forall(j in S diff {0}, k in K, c in C) sum(i in Sprime diff{j}) Y[i,j,k,c] == sum(i in Sprime diff{j}) Y[j,i,k,c];
//   //NB: I needed to remove the depot from S
//   
//   //new
//   forall(i in Sprime, j in Sprime diff {i}, k in K, c in C) Y[i,j,k,c] <= U[k];
//   
//   // (4) Cycle finish (recharge) node be the same as the start of the next cycle. 
//   forall(k in K, c in C) sum(i in S, j in R) Y[i,j,k,c] <= 1;
//   // (5) Not enough time to go from i to recharge r then to j***
//   forall(k in K, r in R, c in C diff {1}) sum(i in S) Y[i,r,k,(c-1)] == sum(j in S) Y[r,j,k,c];
//   // (6) Not enough time to do i then j with the same bus
////   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,c-1] + Y[r,j,k,c]) <= (2*PHI[i,r,j]);
//	forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,(c-1)] + Y[r,j,k,c]) <= phi[i, j, r] + 1;
//	//TODO: reactivate
////   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,(c-1)] + Y[r,j,k,c]) <= PHI[i, j] + 1;
//   //NB: 2*PHI was wrong because we are considering all pairs i,j;
//   //if we have i->r->j with PHIij=1, all is good, but there will also be a constraint checking other combinations
//   //e.g. k->r->j which may have PHIkj=0. This will prevent going to j from r even for other valid combinations
//   //since RHS for pair k,j will be <= 0, it will for Yrjkc=0 regardless of anythings
//   
//   //debugging...
//   //Y[0,1,1,1] == 1;
//   //Y[1,3,1,1] == 1;
//   //Y[3,4,1,1] == 1;
//   //Y[4,2,1,2] == 1;
//   //Y[2,0,1,2] == 1;
//   //Y[3,4,1,1] + Y[4,2,1,2] <= 2;				//this worked
//   //Y[3,4,1,1] + Y[4,2,1,2] <= 2 * PHI[3,2];		//so did this
//   
////   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,c-1] + Y[r,j,k,c]) <= (2*rPHI[(r-NTrips-1)+i,j]);
//   // (7) Feasible constraint
//   forall(i in S, j in S diff {i}) sum(k in K, c in C) Y[i,j,k,c] <= gamma[i,j];
////   forall(i in S, j in S diff {i}) sum(k in K, c in {1}) Y[i,j,k,c] <= gamma[i,j];
//   // (8) Energy constraint
//   forall(k in K, c in C) sum(i in Sprime, j in Sprime diff{i}) delta[i,j] * Y[i,j,k,c] <= MAX_DURATION;
//   // (9) Continue Recharging Cycle (NB: REMOVED - THIS WILL CREATE INFEASIBILITY)
//   //forall(k in K, i in Sprime, j in Sprime diff{i}, c in C diff{1}) Y[i,j,k,c] <= Y[i,j,k,c-1];
//   // (10) Constraint to find only feasible solutions
//   //forall(k in K) sum(j in S diff{0}) Y[0,j,k,1] == U[k];
//   
//   //new: need these to force depot at start and end, other constraint sdidn't cut it'
//   forall(k in K, c in C diff{1}, j in Sprime) Y[0,j,k,c] == 0;
//   forall(k in K, c in C diff{1}, i in Sprime, j in Sprime diff{i}) Y[i,j,k,c] <= 1 - sum(h in Sprime) Y[h,0,k,c-1];
//   forall(k in K, c in C, r in R) Y[0,r,k,c] + Y[r,0,k,c] == 0;
//   
//   // (11) Continuous Sequence
//   forall(k in K diff{1}) U[k] <= U[k-1];
//   
////   // Cannot exceed the maximum number of visits to recharge station
////   // (1) the out-flow must be one (supply constraint)
////   forall(j in S diff {0}) sum(i in Sprime diff{j}, k in K, c in C) Y[i,j,k,c] == 1;
////   // (2) the in-flow must be one (demand constraint)
////   forall(i in S diff {0}) sum(j in Sprime diff{i}, k in K, c in C) Y[i,j,k,c] == 1;
////   // (3) the number of in-flow and out-flow must be the same
////   forall(j in S diff {0}, k in K, c in C) sum(i in Sprime diff{j}) Y[i,j,k,c] == sum(i in Sprime diff{j}) Y[j,i,k,c];
////   //NB: I needed to remove the depot from S
////   
////   //new
////   forall(i in Sprime, j in Sprime diff {i}, k in K, c in C) Y[i,j,k,c] <= U[k];
////   
////   // (4) Cycle finish (recharge) node be the same as the start of the next cycle. 
////   forall(k in K, c in C) sum(i in S, j in R) Y[i,j,k,c] <= 1;
////   // (5) Not enough time to go from i to recharge r then to j***
////   forall(k in K, r in R, c in C diff {1}) sum(i in S) Y[i,r,k,(c-1)] == sum(j in S) Y[r,j,k,c];
////   // (6) Not enough time to do i then j with the same bus
//////   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,c-1] + Y[r,j,k,c]) <= (2*PHI[i,r,j]);
////	forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,(c-1)] + Y[r,j,k,c]) <= phi[i,j,r] + 1;
//////	forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,(c-1)] + Y[r,j,k,c]) <= PHI[(r-NTrips-1)+i, j] + 1;
////	//TODO: reactivate
//////   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,(c-1)] + Y[r,j,k,c]) <= PHI[i, j] + 1;
////   //NB: 2*PHI was wrong because we are considering all pairs i,j;
////   //if we have i->r->j with PHIij=1, all is good, but there will also be a constraint checking other combinations
////   //e.g. k->r->j which may have PHIkj=0. This will prevent going to j from r even for other valid combinations
////   //since RHS for pair k,j will be <= 0, it will for Yrjkc=0 regardless of anythings
////   
////   //debugging...
////   //Y[0,1,1,1] == 1;
////   //Y[1,3,1,1] == 1;
////   //Y[3,4,1,1] == 1;
////   //Y[4,2,1,2] == 1;
////   //Y[2,0,1,2] == 1;
////   //Y[3,4,1,1] + Y[4,2,1,2] <= 2;				//this worked
////   //Y[3,4,1,1] + Y[4,2,1,2] <= 2 * PHI[3,2];		//so did this
////   
//////   forall(k in K, r in R, i in S, j in S diff{i}, c in C diff{1}) (Y[i,r,k,c-1] + Y[r,j,k,c]) <= (2*rPHI[(r-NTrips-1)+i,j]);
////   // (7) Feasible constraint
////   forall(i in S, j in S diff {i}) sum(k in K, c in C) Y[i,j,k,c] <= gamma[i,j];
//////   forall(i in S, j in S diff {i}) sum(k in K, c in {1}) Y[i,j,k,c] <= gamma[i,j];
////   // (8) Energy constraint
////   forall(k in K, c in C) sum(i in Sprime, j in Sprime diff{i}) delta[i,j] * Y[i,j,k,c] <= MAX_DURATION;
////   // (9) Continue Recharging Cycle (NB: REMOVED - THIS WILL CREATE INFEASIBILITY)
////   //forall(k in K, i in Sprime, j in Sprime diff{i}, c in C diff{1}) Y[i,j,k,c] <= Y[i,j,k,c-1];
////   // (10) Constraint to find only feasible solutions
////   //forall(k in K) sum(j in S diff{0}) Y[0,j,k,1] == U[k];
////   
////   //new: need these to force depot at start and end, other constraint sdidn't cut it'
////   forall(k in K, c in C diff{1}, j in Sprime) Y[0,j,k,c] == 0;
////   forall(k in K, c in C diff{1}, i in Sprime, j in Sprime diff{i}) Y[i,j,k,c] <= 1 - sum(h in Sprime) Y[h,0,k,c-1];
////   forall(k in K, c in C, r in R) Y[0,r,k,c] + Y[r,0,k,c] == 0;
////   
////   // (11) Continuous Sequence
////   forall(k in K diff{1}) U[k] <= U[k-1];
// } 