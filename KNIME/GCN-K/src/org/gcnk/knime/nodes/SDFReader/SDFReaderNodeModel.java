package org.gcnk.knime.nodes.SDFReader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.knime.core.data.DataCell;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.DataColumnSpecCreator;
import org.knime.core.data.DataRow;
import org.knime.core.data.DataTableSpec;
import org.knime.core.data.RowKey;
import org.knime.core.data.def.DefaultRow;
import org.knime.core.data.def.StringCell;
import org.knime.core.node.BufferedDataContainer;
import org.knime.core.node.BufferedDataTable;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.defaultnodesettings.SettingsModelIntegerBounded;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.gcnk.knime.nodes.Activator;
import org.gcnk.knime.nodes.GCNNodeModel;

/**
 * This is the model implementation of SDFReader.
 * Read labes from CSV file.
 *
 * @author org.gcnk
 */
public class SDFReaderNodeModel extends GCNNodeModel {
    
    // the logger instance
    private static final NodeLogger logger = NodeLogger
            .getLogger(SDFReaderNodeModel.class);
        
    /** the settings key which is used to retrieve and 
        store the settings (from the dialog or from a settings file)    
       (package visibility to be usable from the dialog). */
	static final String CFGKEY_SDF_FILE = "SDFFile";
	static final String CFGKEY_WORK_DIR = "WorkingDirectory";
	static final String CFGKEY_ATOM_NUM_LIMIT = "AtomNumLimit";

	static final Integer DEFAULT_ATOM_NUM_LIMIT = 70;

    // example value: the models count variable filled from the dialog 
    // and used in the models execution method. The default components of the
    // dialog work with "SettingsModels".

	private final SettingsModelString m_sdfFile = new SettingsModelString(CFGKEY_SDF_FILE, "");
    private final SettingsModelString m_workDir = new SettingsModelString(CFGKEY_WORK_DIR, "");
    private final SettingsModelIntegerBounded m_atomNumLimit =
    		new SettingsModelIntegerBounded(
                        CFGKEY_ATOM_NUM_LIMIT,
                        DEFAULT_ATOM_NUM_LIMIT,
                        1, Integer.MAX_VALUE);

    /**
     * Constructor for the node model.
     */
    protected SDFReaderNodeModel() {
    
        // TODO one incoming port and one outgoing port is assumed
        super(0, 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData,
            final ExecutionContext exec) throws Exception {

        final String scriptPath = Activator.getFile("org.gcnk.knime.nodes", "/py/prep_mol_info.py").getAbsolutePath();
        
        String pythonPath = System.getenv("GCNK_PYTHON_PATH");
        String sdfFile = m_sdfFile.getStringValue();
        
        logger.info("SDF File: " + sdfFile);
        
        String workDir = m_workDir.getStringValue();
        if( workDir == null || workDir.isEmpty() ) {
        	workDir = new File(sdfFile).getParent();
        }
        String logFile = workDir + "/SDFReader.log";
        String outFile = workDir + "/mol_info.jbl";

        logger.info("Working Directory: " + workDir);
        logger.info("Log File: "          + logFile);
        logger.info("Output File: "       + outFile);
        
        String atomNumLimit = String.valueOf(m_atomNumLimit.getIntValue());
        
        String[] Command = { pythonPath, scriptPath, 
        		"--sdf", sdfFile, 
        		"--atom_num_limit", atomNumLimit, 
        		"--output", outFile};
        
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
        
        // the data table spec of the single output table, 
        // the table will have three columns:
        DataColumnSpec[] allColSpecs = new DataColumnSpec[2];
        allColSpecs[0] = new DataColumnSpecCreator("Log File"                 , StringCell.TYPE).createSpec();
        allColSpecs[1] = new DataColumnSpecCreator("Molecule Information File", StringCell.TYPE).createSpec();
        DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
        // the execution context will provide us with storage capacity, in this
        // case a data container to which we will add rows sequentially
        // Note, this container can also handle arbitrary big data tables, it
        // will buffer to disc if necessary.
        BufferedDataContainer container = exec.createDataContainer(outputSpec);
        // let's add rows to it

        {
	        RowKey key = new RowKey("Files");
	        DataCell[] cells = new DataCell[2];
	        cells[0] = new StringCell(logFile);
	        cells[1] = new StringCell(outFile);
	        DataRow row = new DefaultRow(key, cells);
	        container.addRowToTable(row);
        }
        
        // once we are done, we close the container and return its table
        container.close();
        BufferedDataTable out = container.getTable();
        return new BufferedDataTable[]{out};
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
        CheckFileExistence(m_sdfFile.getStringValue(), "SDF file");
    	
        DataColumnSpec[] allColSpecs = new DataColumnSpec[2];
        allColSpecs[0] = new DataColumnSpecCreator("Log File"                 , StringCell.TYPE).createSpec();
        allColSpecs[1] = new DataColumnSpecCreator("Molecule Information File", StringCell.TYPE).createSpec();
        DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
        
        return new DataTableSpec[]{outputSpec};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {

        // TODO save user settings to the config object.
        
        m_sdfFile.saveSettingsTo(settings);
        m_workDir.saveSettingsTo(settings);
        m_atomNumLimit.saveSettingsTo(settings);
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
        
        m_sdfFile.loadSettingsFrom(settings);
        m_workDir.loadSettingsFrom(settings);
        m_atomNumLimit.loadSettingsFrom(settings);
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

        m_sdfFile.validateSettings(settings);
        m_workDir.validateSettings(settings);
        m_atomNumLimit.validateSettings(settings);
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

