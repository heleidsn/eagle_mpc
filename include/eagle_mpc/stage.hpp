///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, Institut de Robotica i Informatica Industrial (CSIC-UPC)
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef EAGLE_MPC_STAGE_HPP_
#define EAGLE_MPC_STAGE_HPP_

#include <map>

#include "boost/enable_shared_from_this.hpp"

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"

#include "eagle_mpc/trajectory.hpp"
#include "eagle_mpc/utils/params_server.hpp"

#include "eagle_mpc/factory/cost.hpp"
#include "eagle_mpc/factory/diff-action.hpp"
#include "eagle_mpc/factory/contacts.hpp"

namespace eagle_mpc
{
class Trajectory;
class ResidualModelFactory;
class ContactModelFactory;
enum class ResidualModelTypes;
// struct ResidualModelTypes {
//     enum Type {};
// };
enum class ContactModelTypes;

class Stage : public boost::enable_shared_from_this<Stage>
{
    public:
    static boost::shared_ptr<Stage> create(const boost::shared_ptr<Trajectory>& trajectory);
    ~Stage();

    void autoSetup(const std::string&                        path_to_stages,
                   const std::map<std::string, std::string>& stage,
                   const boost::shared_ptr<ParamsServer>&    server,
                   std::size_t                               t_ini);

    void set_t_ini(const std::size_t& t_ini);
    void set_duration(const std::size_t& duration);

    const boost::shared_ptr<Trajectory>&                      get_trajectory() const;
    const boost::shared_ptr<crocoddyl::CostModelSum>&         get_costs() const;
    const boost::shared_ptr<crocoddyl::ContactModelMultiple>& get_contacts() const;

    const std::map<std::string, ResidualModelTypes>& get_cost_types() const;
    const std::map<std::string, ContactModelTypes>&  get_contact_types() const;

    const std::size_t& get_t_ini() const;

    const std::size_t& get_duration() const;
    const bool&        get_is_transition() const;
    const bool&        get_is_terminal() const;
    const std::string& get_name() const;

    private:
    Stage(const boost::shared_ptr<Trajectory>& trajectory);

    boost::shared_ptr<crocoddyl::CostModelSum>         costs_;
    boost::shared_ptr<crocoddyl::ContactModelMultiple> contacts_;
    boost::shared_ptr<Trajectory>                      trajectory_;

    std::map<std::string, ResidualModelTypes> cost_types_;
    std::map<std::string, ContactModelTypes>  contact_types_;

    boost::shared_ptr<ResidualModelFactory> residual_factory_;
    boost::shared_ptr<ContactModelFactory>  contact_factory_;

    std::string name_;
    std::size_t duration_;
    std::size_t t_ini_;

    bool is_terminal_;
    bool is_transition_;
};

}  // namespace eagle_mpc

#endif