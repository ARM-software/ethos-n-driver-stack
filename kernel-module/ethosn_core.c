/*
 *
 * (C) COPYRIGHT 2020-2021 Arm Limited.
 *
 * This program is free software and is provided to you under the terms of the
 * GNU General Public License version 2 as published by the Free Software
 * Foundation, and any use by you of this program is subject to the terms
 * of such GNU licence.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, you can access it online at
 * http://www.gnu.org/licenses/gpl-2.0.html.
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 */

#include "ethosn_device.h"
#include "ethosn_core.h"
#include "ethosn_network.h"

#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/platform_device.h>
#include <linux/iommu.h>
#include <linux/pm_runtime.h>

#define ETHOSN_CORE_DRIVER_NAME    "ethosn-core"
#define ETHOSN_CORE_NUM_MAX        64

#ifdef CONFIG_PM
static bool ethosn_is_sleeping(struct ethosn_core *core)
{
	struct dl1_sysctlr0_r sysctlr0 = { .word = 0 };

	sysctlr0.word =
		ethosn_read_top_reg(core, DL1_RP, DL1_SYSCTLR0);

	return sysctlr0.bits.sleeping;
}

static int ethosn_pm_resume(struct device *dev)
{
	int ret;
	struct ethosn_core *core = dev_get_drvdata(dev);

	if (!core) {
		dev_dbg(dev, "Driver data not found\n");
		ret = -EFAULT;
		goto exit_pm_resume;
	}

	ret = ethosn_reset_and_start_ethosn(core);
	if (ret)
		goto exit_pm_resume;

	ret = mutex_lock_interruptible(&core->mutex);
	if (ret)
		goto exit_pm_resume;

	if (core->current_inference)
		ethosn_schedule_inference(core->current_inference);
	else
		ethosn_schedule_queued_inference(core);

	/* ethosn_schedule_queued_inference modifies current_inference,
	 * put if nothing has been scheduled on this core
	 */
	if (!core->current_inference) {
		pm_runtime_mark_last_busy(core->dev);
		pm_runtime_put(core->dev);
	}

	mutex_unlock(&core->mutex);

exit_pm_resume:
	dev_dbg(dev, "Core pm resume: %d\n", ret);

	return ret;
}

static int ethosn_rpm_resume(struct device *dev)
{
	int ret;
	struct ethosn_core *core = dev_get_drvdata(dev);

	if (!core) {
		dev_dbg(dev, "Driver data not found\n");
		ret = -EFAULT;
		goto exit_rpm_resume;
	}

	ret = ethosn_reset_and_start_ethosn(core);

exit_rpm_resume:
	if (!ret && core->profiling.config.enable_profiling)
		++core->profiling.rpm_resume;

	dev_dbg(dev, "Core rpm resume: %d\n", ret);

	return ret;
}

static int ethosn_pm_suspend_noirq(struct device *dev)
{
	int ret = 0;
	struct ethosn_core *core = dev_get_drvdata(dev);

	if (!core) {
		dev_dbg(dev, "Driver data not found\n");
		ret = -EFAULT;
		goto exit_pm_suspend;
	}

	if (core->current_inference)
		core->current_inference->status = ETHOSN_INFERENCE_SCHEDULED;
	else
		/* Get if nothing was scheduled on this core */
		pm_runtime_get_noresume(core->dev);

	ret = ethosn_reset(core);
exit_pm_suspend:
	dev_dbg(dev, "Core pm suspend: %d\n", ret);

	return ret;
}

static int ethosn_rpm_suspend(struct device *dev)
{
	int ret = 0;
	struct ethosn_core *core = dev_get_drvdata(dev);

	if (!core) {
		dev_dbg(dev, "Driver data not found\n");
		ret = -EFAULT;
		goto exit_rpm_suspend;
	}

	if (!ethosn_is_sleeping(core))
		ret = -EBUSY;

exit_rpm_suspend:
	if (!ret && core->profiling.config.enable_profiling)
		++core->profiling.rpm_suspend;

	dev_dbg(dev, "Core rpm suspend: %d\n", ret);

	return ret;
}

static const struct dev_pm_ops ethosn_pm_ops = {
	.resume        = ethosn_pm_resume,
	.suspend_noirq = ethosn_pm_suspend_noirq,
	SET_RUNTIME_PM_OPS(ethosn_rpm_suspend, ethosn_rpm_resume, NULL)
};
#define ETHOSN_PM_OPS (&ethosn_pm_ops)
#else
#define ETHOSN_PM_OPS (NULL)
#endif  /* CONFIG_PM */

int ethosn_get_autosuspend_delay(void)
{
	return ETHOSN_AUTOSUSPEND_DELAY_MS;
}

/* Exported for use by test module */
EXPORT_SYMBOL(ethosn_get_autosuspend_delay);

static struct ethosn_device *ethosn_driver(struct platform_device *pdev)
{
	struct ethosn_device *ethosn = dev_get_drvdata(pdev->dev.parent);

	return ethosn;
}

static int ethosn_child_pdev_remove(struct platform_device *pdev)
{
	dev_dbg(&pdev->dev, "Removed the ethosn-core device\n");

	return 0;
}

static int ethosn_child_pdev_probe(struct platform_device *pdev)
{
	struct ethosn_device *ethosn = ethosn_driver(pdev);
	int core_id;
	int core_count = of_get_child_count(pdev->dev.parent->of_node);
	int ret = 0;

	dev_info(&pdev->dev, "Probing core\n");

	if (IS_ERR_OR_NULL(ethosn)) {
		dev_err(&pdev->dev, "Invalid parent device driver");

		return -EINVAL;
	}

	if (core_count > ETHOSN_CORE_NUM_MAX) {
		dev_err(&pdev->dev, "Invalid number of cores, max = %d\n",
			ETHOSN_CORE_NUM_MAX);

		return -EINVAL;
	}

	core_id = ethosn->num_cores;

	if (core_id >= core_count) {
		dev_err(&pdev->dev, "Invalid core id enumeration (%d)\n",
			core_id);

		return -EINVAL;
	}

	/* Allocating the core device (ie struct ethosn_core)
	 * Allocated against parent device
	 */
	ethosn->core[core_id] = devm_kzalloc(
		pdev->dev.parent,
		sizeof(struct ethosn_core),
		GFP_KERNEL);

	if (!ethosn->core[core_id])
		return -ENOMEM;

	/* Link child device object */
	ethosn->core[core_id]->dev = &pdev->dev;
	ethosn->core[core_id]->core_id = core_id;
	ethosn->core[core_id]->parent = ethosn;

	pm_runtime_set_autosuspend_delay(&pdev->dev,
					 ETHOSN_AUTOSUSPEND_DELAY_MS);
	pm_runtime_use_autosuspend(&pdev->dev);
	pm_runtime_get_noresume(&pdev->dev);
	pm_runtime_set_active(&pdev->dev);
	pm_runtime_enable(&pdev->dev);

	dev_set_drvdata(&pdev->dev, ethosn->core[core_id]);

	dev_dbg(&pdev->dev, "Core probed\n");

	++ethosn->num_cores;

	return ret;
}

static const struct of_device_id ethosn_child_pdev_match[] = {
	{ .compatible = ETHOSN_CORE_DRIVER_NAME },
	{ /* Sentinel */ },
};

MODULE_DEVICE_TABLE(of, ethosn_child_pdev_match);

static struct platform_driver ethosn_child_pdev_driver = {
	.probe                  = &ethosn_child_pdev_probe,
	.remove                 = &ethosn_child_pdev_remove,
	.driver                 = {
		.name           = ETHOSN_CORE_DRIVER_NAME,
		.owner          = THIS_MODULE,
		.of_match_table = of_match_ptr(ethosn_child_pdev_match),
		.pm             = ETHOSN_PM_OPS,
	},
};

int ethosn_core_platform_driver_register(void)
{
	pr_info("Registering %s", ETHOSN_CORE_DRIVER_NAME);

	return platform_driver_register(&ethosn_child_pdev_driver);
}

void ethosn_core_platform_driver_unregister(void)
{
	pr_info("Unregistering %s", ETHOSN_CORE_DRIVER_NAME);
	platform_driver_unregister(&ethosn_child_pdev_driver);
}
