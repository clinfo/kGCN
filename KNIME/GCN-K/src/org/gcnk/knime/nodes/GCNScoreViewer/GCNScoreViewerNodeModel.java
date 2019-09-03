package org.gcnk.knime.nodes.GCNScoreViewer;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.knime.core.data.DataTableSpec;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.gcnk.knime.nodes.Activator;
import org.gcnk.knime.nodes.GCNNodeModel;

/**
 * This is the model implementation of GCNScoreViewer.
 * Show scores from output of GCNScore
 *
 * @author org.gcnk
 */
public class GCNScoreViewerNodeModel extends GCNNodeModel {
    
    // the logger instance
    private static final NodeLogger logger = NodeLogger
            .getLogger(GCNScoreViewerNodeModel.class);
        
    /** the settings key which is used to retrieve and 
        store the settings (from the dialog or from a settings file)    
       (package visibility to be usable from the dialog). */

    // example value: the models count variable filled from the dialog 
    // and used in the models execution method. The default components of the
    // dialog work with "SettingsModels".
    static final String CFGKEY_PLOT_MULTITASK          = "PlotMultitask";
    static final Boolean DEFAULT_PLOT_MULTITASK         = false;
    private final SettingsModelBoolean m_multitask =
    		new SettingsModelBoolean(
    				CFGKEY_PLOT_MULTITASK,
    				DEFAULT_PLOT_MULTITASK);
    
    /**
     * Constructor for the node model.
     */
    protected GCNScoreViewerNodeModel() {
    
        // TODO one incoming port and one outgoing port is assumed
        super(1, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData,
            final ExecutionContext exec) throws Exception {

        final String scriptPath = Activator.getFile("org.gcnk.knime.nodes", "/py/gcn_score_viewer.py").getAbsolutePath();
        
        String pythonPath = System.getenv("GCNK_PYTHON_PATH");
        
    	String predictionFile = getInPortFile("Prediction File", inData[0]);
        String workDir = new File(predictionFile).getParent();

        String logFile = workDir + "/GCNScoreViewer.log";
        String outDir  = workDir + "/result_predict/";
        ArrayList<String> command  = new ArrayList<String>();
        command.add(pythonPath);
        command.add(scriptPath);
        command.add("--prediction_data");
        command.add( predictionFile);
        command.add("--output");
        command.add(outDir);
        logger.info("Flag: " +String.valueOf(m_multitask.getBooleanValue()));
        if (m_multitask.getBooleanValue()==true){
        	command.add("--plot_multitask");	
        }
        
        String cmd = String.join(" ", command);
        logger.info("COMMAND: " + cmd);

        File dir = new File(workDir);
        final Path log = Paths.get(logFile);
        
        new File(workDir + "/result_predict").mkdir();

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        pb.redirectOutput(log.toFile());
        
    	pb.directory(dir);
    	Process proc = pb.start();
        proc.waitFor();
        
        return new BufferedDataTable[]{};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void reset() {
        // TODO Code executed on reset.
        // Models build during execute are cleared here.
        // Also data handled in load/saveInternals will be erased here.
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected DataTableSpec[] configure(final DataTableSpec[] inSpecs)
            throws InvalidSettingsException {
        
        // TODO: check if user settings are available, fit to the incoming
        // table structure, and the incoming types are feasible for the node
        // to execute. If the node can execute in its current state return
        // the spec of its output data table(s) (if you can, otherwise an array
        // with null elements), or throw an exception with a useful user message
    	CheckPythonPath();

        // check spec with selected column
    	CheckInportFile("Prediction File", inSpecs[0]);
        
        return new DataTableSpec[]{};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {

        // TODO save user settings to the config object.
    	m_multitask.saveSettingsTo(settings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings)
            throws InvalidSettingsException {
            
        // TODO load (valid) settings from the config object.
        // It can be safely assumed that the settings are valided by the 
        // method below.
    	m_multitask.loadSettingsFrom(settings);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void validateSettings(final NodeSettingsRO settings)
            throws InvalidSettingsException {
            
        // TODO check if the settings could be applied to our model
        // e.g. if the count is in a certain range (which is ensured by the
        // SettingsModel).
        // Do not actually set any values of any member variables.
    	m_multitask.validateSettings(settings);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    protected void loadInternals(final File internDir,
            final ExecutionMonitor exec) throws IOException,
            CanceledExecutionException {
        
        // TODO load internal data. 
        // Everything handed to output ports is loaded automatically (data
        // returned by the execute method, models loaded in loadModelContent,
        // and user settings set through loadSettingsFrom - is all taken care 
        // of). Load here only the other internals that need to be restored
        // (e.g. data used by the views).

    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveInternals(final File internDir,
            final ExecutionMonitor exec) throws IOException,
            CanceledExecutionException {
       
        // TODO save internal models. 
        // Everything written to output ports is saved automatically (data
        // returned by the execute method, models saved in the saveModelContent,
        // and user settings saved through saveSettingsTo - is all taken care 
        // of). Save here only the other internals that need to be preserved
        // (e.g. data used by the views).

    }

}

