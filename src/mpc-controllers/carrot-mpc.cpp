#include <algorithm>

#include "multicopter_mpc/mpc-controllers/carrot-mpc.hpp"
#include "multicopter_mpc/utils/log.hpp"

namespace multicopter_mpc {

CarrotMpc::CarrotMpc(const boost::shared_ptr<Trajectory>& trajectory, const std::vector<Eigen::VectorXd>& state_ref,
                     const std::size_t dt_ref, const std::string& yaml_path)
    : MpcAbstract(yaml_path), trajectory_(trajectory) {
  state_ref_ = std::vector<Eigen::VectorXd>(state_ref.size(), robot_state_->zero());
  std::copy(state_ref.begin(), state_ref.end(), state_ref_.begin());
  for (std::size_t i = 0; i < state_ref_.size(); ++i) {
    t_ref_.push_back(dt_ref * i);
  }

  try {
    carrot_weight_ = params_server_->getParam<double>("mpc_controller/carrot_weight");
  } catch (const std::exception& e) {
    MMPC_WARN << "The following key: 'mpc_controller/carrot_weight' has not been found in the parameters server. Set "
                 "to 10.0";
    carrot_weight_ = 10.0;
  }

  try {
    carrot_weight_end_ = params_server_->getParam<double>("mpc_controller/carrot_weight_end");
  } catch (const std::exception& e) {
    MMPC_WARN
        << "The following key: 'mpc_controller/carrot_weight_end' has not been found in the parameters server. Set "
           "to 5.0";
    carrot_weight_end_ = 5.0;
  }

  t_stages_.reserve(trajectory_->get_stages().size());
  for (std::size_t i = 0; i < trajectory_->get_stages().size(); ++i) {
    t_stages_.push_back(trajectory_->get_stages()[i]->get_t_ini());
  }

  createProblem();

  update_vars_.state_ref = robot_state_->zero();
}

CarrotMpc::~CarrotMpc() {}

void CarrotMpc::createProblem() {
  DifferentialActionModelTypes dif_type;
  if (trajectory_->get_has_contact()) {
    dif_type = DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics;
  } else {
    dif_type = DifferentialActionModelTypes::DifferentialActionModelFreeFwdDynamics;
  }

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
        dam = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(robot_state_, actuation, costs);
        break;
      case DifferentialActionModelTypes::DifferentialActionModelContactFwdDynamics:
        MMPC_ERROR << "Carrot with contact has not been implemented";
        break;
    }

    boost::shared_ptr<crocoddyl::ActionModelAbstract> iam;
    double dt_s = double(params_.dt) / 1000.;
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
      solver_ = boost::make_shared<multicopter_mpc::SolverSbFDDP>(problem_, squash_);
      break;
    case SolverTypes::SolverBoxFDDP:
      solver_ = boost::make_shared<crocoddyl::SolverBoxFDDP>(problem_);
      break;
    case SolverTypes::SolverBoxDDP:
      solver_ = boost::make_shared<crocoddyl::SolverBoxDDP>(problem_);
      break;
  }
  solver_callbacks_.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
  solver_->setCallbacks(solver_callbacks_);
}

boost::shared_ptr<crocoddyl::CostModelSum> CarrotMpc::createCosts() const {
  boost::shared_ptr<crocoddyl::CostModelSum> costs =
      boost::make_shared<crocoddyl::CostModelSum>(robot_state_, actuation_->get_nu());

  for (auto stage = trajectory_->get_stages().begin(); stage != trajectory_->get_stages().end(); ++stage) {
    std::string path_to_stage = "stages/" + (*stage)->get_name();
    for (auto ctype = (*stage)->get_cost_types().begin(); ctype != (*stage)->get_cost_types().end(); ++ctype) {
      std::string cost_name = ctype->first;
      CostModelTypes cost_type = ctype->second;
      boost::shared_ptr<crocoddyl::CostModelAbstract> cost =
          cost_factory_->create(path_to_stage + "/costs/" + cost_name + "/", trajectory_->get_params_server(),
                                robot_state_, actuation_->get_nu(), cost_type);
      costs->addCost((*stage)->get_name() + "/" + cost_name, cost,
                     (*stage)->get_costs()->get_costs().at(cost_name)->weight, false);
    }
  }

  boost::shared_ptr<crocoddyl::CostModelState> carrot_cost =
      boost::make_shared<crocoddyl::CostModelState>(robot_state_, robot_state_->zero(), actuation_->get_nu());
  costs->addCost("carrot_state", carrot_cost, carrot_weight_, false);

  costs->get_costs().at("approach/control_reg")->active = true;
  costs->get_costs().at("approach/state_reg")->active = true;
  costs->get_costs().at("approach/state_limits")->active = true;

  return costs;
}

void CarrotMpc::updateProblem(const std::size_t& current_time) {
  computeActiveStage(current_time);
  update_vars_.idx_last_stage = update_vars_.idx_stage;
  for (std::size_t i = 0; i < dif_models_.size(); ++i) {
    update_vars_.node_time = current_time + i * params_.dt;
    computeActiveStage(update_vars_.node_time, update_vars_.idx_last_stage);
    update_vars_.name_stage = trajectory_->get_stages()[update_vars_.idx_stage]->get_name() + "/";
    if (trajectory_->get_has_contact()) {
      updateContactCosts(i);
    } else {
      updateFreeCosts(i, current_time);
    }
    update_vars_.idx_last_stage = update_vars_.idx_stage;
  }
  // std::cout << "This is the current time: " << current_time << std::endl;
}

void CarrotMpc::computeActiveStage(const std::size_t& current_time) {
  update_vars_.idx_stage =
      std::size_t(std::upper_bound(t_stages_.begin(), t_stages_.end(), current_time) - t_stages_.begin()) - 1;
}

void CarrotMpc::computeActiveStage(const std::size_t& current_time, const std::size_t& last_stage) {
  computeActiveStage(current_time);
  if (update_vars_.idx_stage == last_stage + 2) {
    update_vars_.idx_stage -= 1;
  }
}

void CarrotMpc::updateContactCosts(const std::size_t& idx) {}

void CarrotMpc::updateFreeCosts(const std::size_t& idx, const std::size_t& current_time) {
  update_vars_.dif_free =
      boost::static_pointer_cast<crocoddyl::DifferentialActionModelFreeFwdDynamics>(dif_models_[idx]);

  if (!trajectory_->get_stages()[update_vars_.idx_stage]->get_is_transition() || (idx == dif_models_.size() - 1)) {
    update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->active = true;
    computeStateReference(update_vars_.node_time);
    update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->cost->set_reference(update_vars_.state_ref);

    if (current_time <=
            trajectory_->get_stages().back()->get_t_ini() + trajectory_->get_stages().back()->get_duration() &&
        update_vars_.node_time >
            trajectory_->get_stages().back()->get_t_ini() + trajectory_->get_stages().back()->get_duration()) {
      
      std::cout << "It was: " << update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->active
                << std::endl;
      
      // if (last stage node counter > 1 then false carrot)
      update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->active = false;
      
      std::cout << "Node idx: " << idx << std::endl;
      std::cout << "Node time: " << update_vars_.node_time << std::endl;
      std::cout << "Trajectory ending: "
                << trajectory_->get_stages().back()->get_t_ini() + trajectory_->get_stages().back()->get_duration()
                << std::endl;
    } else if (current_time >
                   trajectory_->get_stages().back()->get_t_ini() + trajectory_->get_stages().back()->get_duration() &&
               update_vars_.node_time >
                   trajectory_->get_stages().back()->get_t_ini() + trajectory_->get_stages().back()->get_duration()) {
      update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->active = true;
      update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->weight = carrot_weight_end_;
      update_vars_.dif_free->get_costs()->get_costs().at("approach/state_reg")->active = false;
      // std::cout << "Current time: " << current_time << std::endl;
      // std::cout << "Finish time: "
      //           << trajectory_->get_stages().back()->get_t_ini() + trajectory_->get_stages().back()->get_duration()
      //           << std::endl;
      // std::cout << "node time: " << update_vars_.node_time << std::endl;
    }
  } else {
    update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->active = false;
  }
}

void CarrotMpc::updateFreeCostsTasks(const std::size_t& idx) {
  update_vars_.dif_free =
      boost::static_pointer_cast<crocoddyl::DifferentialActionModelFreeFwdDynamics>(dif_models_[idx]);
  for (auto cost = update_vars_.dif_free->get_costs()->get_costs().begin();
       cost != update_vars_.dif_free->get_costs()->get_costs().end(); cost++) {
    if (cost->first.compare(0, update_vars_.name_stage.size(), update_vars_.name_stage) == 0 ||
        cost->first.compare(0, cost->first.size(), "barrier") == 0) {
      cost->second->active = true;
    } else {
      cost->second->active = false;
    }
  }

  if (idx == dif_models_.size() - 1 && trajectory_->get_stages()[update_vars_.idx_stage]->get_is_transition()) {
    update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->active = true;
    computeStateReference(update_vars_.node_time);
    update_vars_.dif_free->get_costs()->get_costs().at("carrot_state")->cost->set_reference(update_vars_.state_ref);
  }
}

void CarrotMpc::computeStateReference(const std::size_t& time) {
  update_vars_.idx_state = std::size_t(std::upper_bound(t_ref_.begin(), t_ref_.end(), time) - t_ref_.begin());
  if (update_vars_.idx_state >= state_ref_.size()) {
    update_vars_.state_ref = robot_state_->zero();
    update_vars_.state_ref.head(robot_state_->get_nq()) = state_ref_.back().head(robot_state_->get_nq());
    update_vars_.quat_hover = Eigen::Quaterniond(state_ref_.back()(6), 0.0, 0.0, state_ref_.back()(5));
    update_vars_.quat_hover.normalize();
    update_vars_.state_ref(3) = update_vars_.quat_hover.x();
    update_vars_.state_ref(4) = update_vars_.quat_hover.y();
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

const boost::shared_ptr<Trajectory>& CarrotMpc::get_trajectory() const { return trajectory_; }
const std::vector<Eigen::VectorXd>& CarrotMpc::get_state_ref() const { return state_ref_; }
const std::vector<std::size_t>& CarrotMpc::get_t_stages() const { return t_stages_; }
const std::vector<std::size_t>& CarrotMpc::get_t_ref() const { return t_ref_; }

}  // namespace multicopter_mpc
