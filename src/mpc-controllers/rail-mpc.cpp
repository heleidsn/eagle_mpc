///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, Institut de Robotica i Informatica Industrial (CSIC-UPC)
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include "eagle_mpc/mpc-controllers/rail-mpc.hpp"
#include "eagle_mpc/utils/log.hpp"

namespace eagle_mpc
{
RailMpc::RailMpc(const std::vector<Eigen::VectorXd>& state_ref, const std::size_t dt_ref, const std::string& yaml_path)
    : MpcAbstract(yaml_path), use_planner_control_(false)
{   
    std::cout << "==============RailMpc constructor (without control_ref) [Lei He]===============" << std::endl;

    state_ref_ = std::vector<Eigen::VectorXd>(state_ref.size(), robot_state_->zero());
    std::copy(state_ref.begin(), state_ref.end(), state_ref_.begin());
    for (std::size_t i = 0; i < state_ref_.size(); ++i) {
        t_ref_.push_back(dt_ref * i);
    }

    try {
        state_weight_ = params_server_->getParam<double>("mpc_controller/rail_weight");
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_weight' has not been found in the parameters server. Set "
            "to 10.0");
        state_weight_ = 10;
    }

    try {
        state_activation_weights_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_activation_weights"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_activation_weights' has not been found in the parameters "
            "server. Set "
            "to unitary vector");
        state_activation_weights_ = Eigen::VectorXd::Ones(robot_state_->get_ndx());
    }
    if (state_activation_weights_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the state activation weights vector is " +
                           std::to_string(state_activation_weights_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    try {
        control_weight_ = params_server_->getParam<double>("mpc_controller/rail_control_weight");
        std::cout << "control_weight_ = \n" << control_weight_ << std::endl;
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_control_weight' has not been found in the parameters "
            "server. Set "
            "to 1e-1");
        control_weight_ = 1e-1;
    }

    // Control reference will be dynamically updated from solver results
    // Initialize with typical hover control values (can be adjusted based on platform)
    try {
        control_reference_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/control_reference"));
            
        std::cout << "control_reference_ = \n" << control_reference_.transpose() << std::endl;

    } catch (const std::exception& e) {
        EMPC_DEBUG(
            "The following key: 'mpc_controller/control_reference' has not been found in the parameters server. Set "
            "to zero vector");
        control_reference_ = Eigen::VectorXd::Zero(actuation_->get_nu());  // 默认值为零向量
    }
    
    // control_reference_ = Eigen::VectorXd::Constant(actuation_->get_nu(), 0); 

    // Load state limits parameters
    try {
        state_limits_weight_ = params_server_->getParam<double>("mpc_controller/rail_state_limits_weight");
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_weight' has not been found in the parameters server. Set "
            "to 0");
        state_limits_weight_ = 0;
    }

    try {
        state_limits_act_weights_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_state_limits_act_weights"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_act_weights' has not been found in the parameters "
            "server. Set to unitary vector");
        state_limits_act_weights_ = Eigen::VectorXd::Ones(robot_state_->get_ndx());
    }
    if (state_limits_act_weights_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the state limits activation weights vector is " +
                           std::to_string(state_limits_act_weights_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    try {
        state_limits_l_bound_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_state_limits_l_bound"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_l_bound' has not been found in the parameters server. Set "
            "to zero vector");
        state_limits_l_bound_ = Eigen::VectorXd::Zero(robot_state_->get_ndx());
    }
    if (state_limits_l_bound_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the lower limits vector is " +
                           std::to_string(state_limits_l_bound_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    try {
        state_limits_u_bound_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_state_limits_u_bound"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_u_bound' has not been found in the parameters server. Set "
            "to zero vector");
        state_limits_u_bound_ = Eigen::VectorXd::Zero(robot_state_->get_ndx());
    }
    if (state_limits_u_bound_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the upper limits vector is " +
                           std::to_string(state_limits_u_bound_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    createProblem();

    update_vars_.state_ref = robot_state_->zero();
}

RailMpc::RailMpc(const std::vector<Eigen::VectorXd>& state_ref,
                 const std::vector<Eigen::VectorXd>& control_ref,
                 const std::size_t dt_ref,
                 const std::string& yaml_path)
    : MpcAbstract(yaml_path), use_planner_control_(true)
{
    std::cout << "==============RailMpc constructor (with control_ref) [Lei He]===============" << std::endl;

    state_ref_ = std::vector<Eigen::VectorXd>(state_ref.size(), robot_state_->zero());
    std::copy(state_ref.begin(), state_ref.end(), state_ref_.begin());
    
    control_ref_ = std::vector<Eigen::VectorXd>(control_ref.size(), Eigen::VectorXd::Zero(actuation_->get_nu()));
    std::copy(control_ref.begin(), control_ref.end(), control_ref_.begin());
    
    for (std::size_t i = 0; i < state_ref_.size(); ++i) {
        t_ref_.push_back(dt_ref * i);
    }

    std::cout << "Loaded " << control_ref_.size() << " control references from planner" << std::endl;

    try {
        state_weight_ = params_server_->getParam<double>("mpc_controller/rail_weight");
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_weight' has not been found in the parameters server. Set "
            "to 10.0");
        state_weight_ = 10;
    }

    try {
        state_activation_weights_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_activation_weights"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_activation_weights' has not been found in the parameters "
            "server. Set "
            "to unitary vector");
        state_activation_weights_ = Eigen::VectorXd::Ones(robot_state_->get_ndx());
    }
    if (state_activation_weights_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the state activation weights vector is " +
                           std::to_string(state_activation_weights_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    try {
        control_weight_ = params_server_->getParam<double>("mpc_controller/rail_control_weight");
        std::cout << "control_weight_ = " << control_weight_ << std::endl;
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_control_weight' has not been found in the parameters "
            "server. Set "
            "to 1e-1");
        control_weight_ = 1e-1;
    }

    // Initialize control reference from planner
    control_reference_ = Eigen::VectorXd::Zero(actuation_->get_nu());
    if (!control_ref_.empty()) {
        control_reference_ = control_ref_[0];
        std::cout << "Initial control_reference_ from planner = " << control_reference_.transpose() << std::endl;
    }

    // Load state limits parameters
    try {
        state_limits_weight_ = params_server_->getParam<double>("mpc_controller/rail_state_limits_weight");
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_weight' has not been found in the parameters server. Set "
            "to 0");
        state_limits_weight_ = 0;
    }

    try {
        state_limits_act_weights_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_state_limits_act_weights"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_act_weights' has not been found in the parameters "
            "server. Set to unitary vector");
        state_limits_act_weights_ = Eigen::VectorXd::Ones(robot_state_->get_ndx());
    }
    if (state_limits_act_weights_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the state limits activation weights vector is " +
                           std::to_string(state_limits_act_weights_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    try {
        state_limits_l_bound_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_state_limits_l_bound"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_l_bound' has not been found in the parameters server. Set "
            "to zero vector");
        state_limits_l_bound_ = Eigen::VectorXd::Zero(robot_state_->get_ndx());
    }
    if (state_limits_l_bound_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the lower limits vector is " +
                           std::to_string(state_limits_l_bound_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    try {
        state_limits_u_bound_ = converter<Eigen::VectorXd>::convert(
            params_server_->getParam<std::string>("mpc_controller/rail_state_limits_u_bound"));
    } catch (const std::exception& e) {
        EMPC_INFO(
            "The following key: 'mpc_controller/rail_state_limits_u_bound' has not been found in the parameters server. Set "
            "to zero vector");
        state_limits_u_bound_ = Eigen::VectorXd::Zero(robot_state_->get_ndx());
    }
    if (state_limits_u_bound_.size() != robot_state_->get_ndx()) {
        std::runtime_error("RailMPC: the dimension for the upper limits vector is " +
                           std::to_string(state_limits_u_bound_.size()) + ", should be " +
                           std::to_string(robot_state_->get_ndx()));
    }

    createProblem();

    update_vars_.state_ref = robot_state_->zero();
}

RailMpc::~RailMpc() {}

void RailMpc::createProblem()
{
    DifferentialActionModelTypes dif_type;
    dif_type = DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics;

    boost::shared_ptr<crocoddyl::ActuationModelAbstract> actuation;
    if (params_.solver_type == SolverTypes::SolverSbFDDP) {
        actuation = actuation_squash_;
    } else {
        actuation = actuation_;
    }

    for (std::size_t i = 0; i < params_.knots; ++i) {
        boost::shared_ptr<crocoddyl::CostModelSum> costs = createCosts();

        boost::shared_ptr<crocoddyl::DifferentialActionModelAbstract> dam;
        switch (dif_type) {
            case DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics:
                dam = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(robot_state_, actuation,
                                                                                            costs);
                break;
            case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics:
                EMPC_ERROR("Carrot with contact has not been implemented");
                break;
        }

        boost::shared_ptr<crocoddyl::ActionModelAbstract> iam;
        double                                            dt_s = double(params_.dt) / 1000.;
        switch (params_.integrator_type) {
            case IntegratedActionModelTypes::IntegratedActionModelEuler:
                iam = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dam, dt_s);
                break;

            case IntegratedActionModelTypes::IntegratedActionModelRK4:
                iam = boost::make_shared<crocoddyl::IntegratedActionModelRK4>(dam, dt_s);
                break;
        }
        iam->set_u_lb(platform_params_->u_lb);
        iam->set_u_ub(platform_params_->u_ub);

        dif_models_.push_back(dam);
        int_models_.push_back(iam);
    }

    problem_ = boost::make_shared<crocoddyl::ShootingProblem>(
        robot_state_->zero(),
        std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>(int_models_.begin(), int_models_.end() - 1),
        int_models_.back());

    switch (params_.solver_type) {
        case SolverTypes::SolverSbFDDP:
            solver_ = boost::make_shared<eagle_mpc::SolverSbFDDP>(problem_, squash_);
            break;
        case SolverTypes::SolverBoxFDDP:
            solver_ = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem_);
            break;
        case SolverTypes::SolverBoxDDP:
            solver_ = boost::make_shared<crocoddyl::SolverBoxDDP>(problem_);
            break;
    }

    if (params_.callback) {
        solver_callbacks_.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
        solver_->setCallbacks(solver_callbacks_);
    }
}

boost::shared_ptr<crocoddyl::CostModelSum> RailMpc::createCosts() const
{
    boost::shared_ptr<crocoddyl::CostModelSum> costs =
        boost::make_shared<crocoddyl::CostModelSum>(robot_state_, actuation_->get_nu());

    // Calculate dt scaling factor (convert from milliseconds to seconds)
    double dt_s = double(params_.dt) / 1000.0;
    
    // Scale state activation weights by dt
    Eigen::VectorXd scaled_state_activation_weights = state_activation_weights_ * dt_s;
    
    boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> activation =
        boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(state_activation_weights_);
    boost::shared_ptr<crocoddyl::ResidualModelState> rail_residual =
        boost::make_shared<crocoddyl::ResidualModelState>(robot_state_, robot_state_->zero(), actuation_->get_nu());
    boost::shared_ptr<crocoddyl::CostModelResidual> rail_cost =
        boost::make_shared<crocoddyl::CostModelResidual>(robot_state_, activation, rail_residual);
    
    // Scale state weight by dt
    double scaled_state_weight = state_weight_ * dt_s;
    costs->addCost("rail_state", rail_cost, scaled_state_weight, true);

    boost::shared_ptr<crocoddyl::ResidualModelControl> control_residual =
        boost::make_shared<crocoddyl::ResidualModelControl>(robot_state_, actuation_->get_nu());

    // Control reference will be dynamically updated in updateFreeCosts()
    // Initialize with current control reference (hover control)
    control_residual->set_reference(control_reference_);

    boost::shared_ptr<crocoddyl::CostModelResidual> control_cost =
        boost::make_shared<crocoddyl::CostModelResidual>(robot_state_, control_residual);
    
    // Scale control weight by dt
    double scaled_control_weight = control_weight_ * dt_s;
    costs->addCost("control", control_cost, scaled_control_weight, true);

    // Add state limits cost
    crocoddyl::ActivationBounds state_limit_bounds(state_limits_l_bound_, state_limits_u_bound_, 1);
    
    // Scale state limits activation weights by dt
    Eigen::VectorXd scaled_state_limits_act_weights = state_limits_act_weights_ * dt_s;
    
    boost::shared_ptr<crocoddyl::ActivationModelWeightedQuadraticBarrier> state_limits_activation =
        boost::make_shared<crocoddyl::ActivationModelWeightedQuadraticBarrier>(state_limit_bounds,
                                                                               scaled_state_limits_act_weights);
    boost::shared_ptr<crocoddyl::ResidualModelState> state_limit_residual =
        boost::make_shared<crocoddyl::ResidualModelState>(robot_state_, robot_state_->zero(), actuation_->get_nu());
    boost::shared_ptr<crocoddyl::CostModelResidual> state_limit_cost =
        boost::make_shared<crocoddyl::CostModelResidual>(robot_state_, state_limits_activation, state_limit_residual);
    
    // Scale state limits weight by dt
    double scaled_state_limits_weight = state_limits_weight_ * dt_s;
    costs->addCost("state_limits", state_limit_cost, scaled_state_limits_weight, true);
    
    EMPC_DEBUG("RailMpc: Applied dt-based weight scaling (dt = ", params_.dt, "ms = ", dt_s, "s)");
    EMPC_DEBUG("  state_weight: ", state_weight_, " -> ", scaled_state_weight);
    EMPC_DEBUG("  control_weight: ", control_weight_, " -> ", scaled_control_weight);
    EMPC_DEBUG("  state_limits_weight: ", state_limits_weight_, " -> ", scaled_state_limits_weight);

    return costs;
}

void RailMpc::updateProblem(const std::size_t& current_time)
{
    for (std::size_t i = 0; i < dif_models_.size(); ++i) {
        update_vars_.node_time = current_time + i * params_.dt;
        // if (trajectory_->get_has_contact()) {
        //   updateContactCosts(i);
        // } else {
        updateFreeCosts(i);
        // }
    }
}

void RailMpc::updateContactCosts(const std::size_t& idx) {}

void RailMpc::updateFreeCosts(const std::size_t& idx)
{
    update_vars_.dif_free =
        boost::static_pointer_cast<crocoddyl::DifferentialActionModelFreeFwdDynamics>(dif_models_[idx]);

    computeStateReference(update_vars_.node_time);
    boost::static_pointer_cast<crocoddyl::ResidualModelState>(
        update_vars_.dif_free->get_costs()->get_costs().at("rail_state")->cost->get_residual())
        ->set_reference(update_vars_.state_ref);
        
    // Update control reference based on mode
    if (use_planner_control_) {
        // Use planner's control reference
        computeControlReference(update_vars_.node_time);
    } else {
        // Use solver solution (warm-start from previous iteration)
        if (solver_->get_us().size() > idx) {
            control_reference_ = solver_->get_us()[idx];
        }
    }
    
    // Update control cost with new reference
    boost::static_pointer_cast<crocoddyl::ResidualModelControl>(
        update_vars_.dif_free->get_costs()->get_costs().at("control")->cost->get_residual())
        ->set_reference(control_reference_);
}

void RailMpc::computeStateReference(const std::size_t& time)
{
    update_vars_.idx_state = std::size_t(std::upper_bound(t_ref_.begin(), t_ref_.end(), time) - t_ref_.begin());
    if (update_vars_.idx_state >= state_ref_.size()) {
        update_vars_.state_ref                        = robot_state_->zero();
        update_vars_.state_ref.head(robot_model_->nq) = state_ref_.back().head(robot_model_->nq);           // 保持最后时刻的位置和姿态
        update_vars_.quat_hover = Eigen::Quaterniond(state_ref_.back()(6), 0.0, 0.0, state_ref_.back()(5)); // 保持yaw角度不变，roll和pitch为0
        update_vars_.quat_hover.normalize();
        update_vars_.state_ref(5) = update_vars_.quat_hover.z();
        update_vars_.state_ref(6) = update_vars_.quat_hover.w();
    } else {
        update_vars_.alpha = (time - t_ref_[update_vars_.idx_state - 1]) /
                             (t_ref_[update_vars_.idx_state] - t_ref_[update_vars_.idx_state - 1]);
        update_vars_.state_ref.head(robot_model_->nq) =
            pinocchio::interpolate(*robot_model_, state_ref_[update_vars_.idx_state - 1].head(robot_model_->nq),
                                   state_ref_[update_vars_.idx_state].head(robot_model_->nq), update_vars_.alpha);
        update_vars_.state_ref.tail(robot_model_->nv) =
            state_ref_[update_vars_.idx_state - 1].tail(robot_model_->nv) +
            update_vars_.alpha * (state_ref_[update_vars_.idx_state].tail(robot_model_->nv) -
                                  state_ref_[update_vars_.idx_state - 1].tail(robot_model_->nv));
    }
}

void RailMpc::computeControlReference(const std::size_t& time)
{
    if (control_ref_.empty()) {
        // No control reference available, use zero or hover control
        control_reference_ = Eigen::VectorXd::Zero(actuation_->get_nu());
        return;
    }

    std::size_t idx_control = std::size_t(std::upper_bound(t_ref_.begin(), t_ref_.end(), time) - t_ref_.begin());
    
    if (idx_control >= control_ref_.size()) {
        // Beyond the reference trajectory, use the last control
        control_reference_ = control_ref_.back();
    } else if (idx_control == 0) {
        // Before the reference trajectory starts, use the first control
        control_reference_ = control_ref_[0];
    } else {
        // Interpolate between control references
        // For control, we use simple linear interpolation (not quaternion interpolation like state)
        double alpha = (time - t_ref_[idx_control - 1]) /
                      (t_ref_[idx_control] - t_ref_[idx_control - 1]);
        control_reference_ = control_ref_[idx_control - 1] +
                            alpha * (control_ref_[idx_control] - control_ref_[idx_control - 1]);
    }
}

const std::vector<Eigen::VectorXd>& RailMpc::get_state_ref() const { return state_ref_; }
const std::vector<std::size_t>&     RailMpc::get_t_ref() const { return t_ref_; }

}  // namespace eagle_mpc
