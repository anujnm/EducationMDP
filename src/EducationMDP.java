import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.Policy;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.commonpolicies.GreedyQPolicy;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.statehashing.DiscreteStateHashFactory;
import burlap.domain.singleagent.graphdefined.GraphDefinedDomain;
import burlap.oomdp.auxiliary.DomainGenerator;
import burlap.oomdp.auxiliary.common.NullTermination;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class EducationMDP {

    DomainGenerator				dg;
    Domain						domain;
    State						initState;
    RewardFunction				rf;
    TerminalFunction			tf;
    DiscreteStateHashFactory	hashFactory;

    int numStates;

    public EducationMDP(double p1, double p2, double p3, double p4) {
        this.numStates = 6;
        this.dg = new GraphDefinedDomain(numStates);

        //actions from initial state 0
        ((GraphDefinedDomain) this.dg).setTransition(0, 0, 1, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(0, 1, 2, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(0, 2, 3, 1.);

        //transitions from action "a" outcome state
        ((GraphDefinedDomain) this.dg).setTransition(1, 0, 1, 1.);

        //transitions from action "b" outcome state
        ((GraphDefinedDomain) this.dg).setTransition(2, 0, 4, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(4, 0, 2, 1.);

        //transitions from action "c" outcome state
        ((GraphDefinedDomain) this.dg).setTransition(3, 0, 5, 1.);
        ((GraphDefinedDomain) this.dg).setTransition(5, 0, 5, 1.);

        this.domain = this.dg.generateDomain();
        this.initState = GraphDefinedDomain.getState(this.domain,0);
        this.rf = new FourParamRF(p1,p2,p3,p4);
        this.tf = new NullTermination();
        this.hashFactory = new DiscreteStateHashFactory();
    }

    public static class FourParamRF implements RewardFunction {
        double p1;
        double p2;
        double p3;
        double p4;

        public FourParamRF(double p1, double p2, double p3, double p4) {
            this.p1 = p1;
            this.p2 = p2;
            this.p3 = p3;
            this.p4 = p4;
        }

        @Override
        public double reward(State s, GroundedAction a, State sprime) {
            int sid = GraphDefinedDomain.getNodeId(s);
            double r;

            if( sid == 0 || sid == 3 ) { // initial state or c1
                r = 0;
            }
            else if( sid == 1 ) { // a
                r = this.p1;
            }
            else if( sid == 2 ) { // b1
                r = this.p2;
            }
            else if( sid == 4 ) { // b2
                r = this.p3;
            }
            else if( sid == 5 ) { // c2
                r = this.p4;
            }
            else {
                throw new RuntimeException("Unknown state: " + sid);
            }

            return r;
        }
    }

    private ValueIteration computeValue(double gamma) {
        ValueIteration vi = null;
        for (int numIterations = 1; numIterations <= 100; numIterations += 1) {
            long startTime = System.nanoTime();
            double maxDelta = 0.0001;
            int maxIterations = 1000;
            vi = new ValueIteration(this.domain, this.rf, this.tf, gamma,
                    this.hashFactory, maxDelta, maxIterations);
            vi.planFromState(this.initState);
            AnalysisAggregator.addMillisecondsToFinishValueIteration((int) (System.nanoTime() - startTime) / 1000000);
            System.out.println("VI time: " + (System.nanoTime() - startTime) / 1000000);
        }
        AnalysisAggregator.printValueIterationTimeResults();
        return vi;
    }

    private PolicyIteration computePolicy(double gamma) {
        PolicyIteration pi = null;
        for (int numIterations = 1; numIterations <= 100; numIterations += 1) {
            long startTime = System.nanoTime();
            double maxDelta = 0.0001;
            int maxEvaluationIterations = 1;
            int maxIterations = 1000;
            pi = new PolicyIteration(this.domain, this.rf, this.tf, gamma,
                    this.hashFactory, maxDelta, maxEvaluationIterations, maxIterations);
            pi.planFromState(this.initState);
            AnalysisAggregator.addMillisecondsToFinishPolicyIteration((int) (System.nanoTime() - startTime) / 1000000);
            System.out.println("VI time: " + (System.nanoTime() - startTime) / 1000000);
        }
        AnalysisAggregator.printPolicyIterationTimeResults();
        return pi;
    }

    private QLearning runQLearning(double gamma) {
        QLearning agent = null;
        Policy p = null;
        EpisodeAnalysis ea = null;
        agent = new QLearning(
                domain,
                this.rf,
                this.tf,
                0.99,
                this.hashFactory,
                0.99, 0.99);
        agent.runLearningEpisodeFrom(this.initState, 1000);
        //agent.plannerInit(domain, this.rf, this.tf, gamma, this.hashFactory);
        //agent.planFromState(this.initState);

        return agent;

    }

    public String bestFirstAction(double gamma) {

//        ValueIteration iteration = computeValue(gamma);

        PolicyIteration iteration = computePolicy(gamma);

        QLearning ql = runQLearning(gamma);

        double[] P = new double[this.numStates];
        for (int i = 0; i < numStates; i++) {
            State s = GraphDefinedDomain.getState(this.domain, i);
            P[i] = iteration.value(s);
        }
        State initialState = GraphDefinedDomain.getState(this.domain, 0);

        Policy p = new GreedyQPolicy(iteration.getCopyOfValueFunction());
        EpisodeAnalysis ea = p.evaluateBehavior(initialState, rf, tf, 100);
        System.out.println("Number of steps: " + ea.numTimeSteps());
        System.out.println("Reward: " + calcRewardInEpisode(ea));

        String actionName = null;
        if (P[1] >= P[2] && P[1] >= P[3]) {
            actionName = "action a";
        } else if (P[2] >= P[1] && P[2] >= P[3]) {
            actionName = "action b";
        } else if (P[3] >= P[2] && P[3] >= P[1]) {
            actionName = "action c";
        }
        AnalysisAggregator.printPolicyIterationResults();
        return actionName;
    }

    public double calcRewardInEpisode(EpisodeAnalysis ea) {
        double myRewards = 0;

        //sum all rewards
        for (int i = 0; i < ea.rewardSequence.size(); i++) {
            myRewards += ea.rewardSequence.get(i);
        }
        return myRewards;
    }

    public static void main(String[] args) {
        double p1 = 4.;
        double p2 = 3.;
        double p3 = 6.;
        double p4 = 5.;
        FirstMDP mdp = new FirstMDP(p1,p2,p3,p4);

        double gamma = 0.89;
        System.out.println("Best initial action: " + mdp.bestFirstAction(gamma));
    }
}
