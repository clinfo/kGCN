package org.gcnk.knime.nodes.GCNGraphViewer;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.knime.core.data.DataTableSpec;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.gcnk.knime.nodes.Activator;
import org.gcnk.knime.nodes.GCNNodeModel;

/**
 * This is the model implementation of GCNGraphViewer.
 * Display graphically the contribution of each atom of each test compound in the test set to the predicted value on the compound from the output of the GCNVisualizer
 *
 * @author org.gcnk
 */
public class GCNGraphViewerNodeModel extends GCNNodeModel {
    
    // the logger instance
    private static final NodeLogger logger = NodeLogger
            .getLogger(GCNGraphViewerNodeModel.class);
        
    /** the settings key which is used to retrieve and 
        store the settings (from the dialog or from a settings file)    
       (package visibility to be usable from the dialog). */

    // example value: the models count variable filled from the dialog 
    // and used in the models execution method. The default components of the
    // dialog work with "SettingsModels".

    /**
     * Constructor for the node model.
     */
    protected GCNGraphViewerNodeModel() {
    
        // TODO one incoming port and one outgoing port is assumed
        super(1, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData,
            final ExecutionContext exec) throws Exception {

        final String scriptPath = Activator.getFile("org.gcnk.knime.nodes", "/py/graph_viewer.py").getAbsolutePath();
        
        String pythonPath = System.getenv("GCNK_PYTHON_PATH");
        
    	String visualizeDir = getInPortFileNoCheck("Visualization Directory", inData[0]);
        String workDir = visualizeDir;

        String logFile = workDir + "/GCNGraphViewer.log";
        
        String[] Command = { pythonPath, scriptPath, 
        		"--directory", visualizeDir};
        
        String cmd = String.join(" ", Command);
        logger.info("COMMAND: " + cmd);

        File dir = new File(workDir);
        final Path log = Paths.get(logFile);
        
        ProcessBuilder pb = new ProcessBuilder(Command);
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
    	CheckInportFile("Visualization Directory", inSpecs[0]);
        
        return new DataTableSpec[]{};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {

        // TODO save user settings to the config object.
        
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

