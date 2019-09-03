package org.gcnk.knime.nodes.GCNDatasetSplitter;

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
import org.knime.core.node.defaultnodesettings.SettingsModelDoubleBounded;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.gcnk.knime.nodes.Activator;
import org.gcnk.knime.nodes.GCNNodeModel;

/**
 * This is the model implementation of GCNDatasetSplitter.
 * Split dataset into two parts.
 *
 * @author org.gcnk
 */
public class GCNDatasetSplitterNodeModel extends GCNNodeModel {
    
    // the logger instance
    private static final NodeLogger logger = NodeLogger
            .getLogger(GCNDatasetSplitterNodeModel.class);
        
    /** the settings key which is used to retrieve and 
        store the settings (from the dialog or from a settings file)    
       (package visibility to be usable from the dialog). */
	static final String CFGKEY_RATIO = "Ratio";
	static final Double  DEFAULT_RATIO        = 0.9;

    // example value: the models count variable filled from the dialog 
    // and used in the models execution method. The default components of the
    // dialog work with "SettingsModels".
    private final SettingsModelDoubleBounded m_ratio =
    		new SettingsModelDoubleBounded(
    				CFGKEY_RATIO,
    				DEFAULT_RATIO,
    				0.0, 1.0);
    

    /**
     * Constructor for the node model.
     */
    protected GCNDatasetSplitterNodeModel() {
    
        // TODO one incoming port and one outgoing port is assumed
        super(1, 2);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData,
            final ExecutionContext exec) throws Exception {

        final String scriptPath = Activator.getFile("org.gcnk.knime.nodes", "/py/split_dataset.py").getAbsolutePath();
        
        String pythonPath = System.getenv("GCNK_PYTHON_PATH");
        
    	// Get DatasetFile
    	String datasetFile = getInPortFile("Dataset File", inData[0]);

        String workDir = new File(datasetFile).getParent();

        String logFile = workDir + "/GCNDatasetSplitter.log";
        String outFile = datasetFile.substring(0, datasetFile.lastIndexOf('.'));
        String outFile1 = outFile + "_split1.jbl";
        String outFile2 = outFile + "_split2.jbl";
        String ratio = String.valueOf(m_ratio.getDoubleValue());
        String[] Command = { pythonPath, scriptPath, 
        		"--dataset", datasetFile, 
        		"--output1", outFile1, 
        		"--output2", outFile2, 
        		"--ratio"  , ratio};
        
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
        allColSpecs[0] = new DataColumnSpecCreator("Log File"    , StringCell.TYPE).createSpec();
        allColSpecs[1] = new DataColumnSpecCreator("Dataset File", StringCell.TYPE).createSpec();
        DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
        // the execution context will provide us with storage capacity, in this
        // case a data container to which we will add rows sequentially
        // Note, this container can also handle arbitrary big data tables, it
        // will buffer to disc if necessary.

        BufferedDataContainer container1 = exec.createDataContainer(outputSpec);
        {
	        RowKey key = new RowKey("Files");
	        DataCell[] cells = new DataCell[2];
	        cells[0] = new StringCell(logFile);
	        cells[1] = new StringCell(outFile1);
	        DataRow row = new DefaultRow(key, cells);
	        container1.addRowToTable(row);
            container1.close();
        }
        BufferedDataTable out1 = container1.getTable();

        BufferedDataContainer container2 = exec.createDataContainer(outputSpec);
        {
	        RowKey key = new RowKey("Files");
	        DataCell[] cells = new DataCell[2];
	        cells[0] = new StringCell(logFile);
	        cells[1] = new StringCell(outFile2);
	        DataRow row = new DefaultRow(key, cells);
	        container2.addRowToTable(row);
            container2.close();
        }
        BufferedDataTable out2 = container2.getTable();
        
        return new BufferedDataTable[]{out1, out2};
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
    	CheckInportFile("Dataset File", inSpecs[0]);
        
        DataColumnSpec[] allColSpecs = new DataColumnSpec[2];
        allColSpecs[0] = new DataColumnSpecCreator("Log File"    , StringCell.TYPE).createSpec();
        allColSpecs[1] = new DataColumnSpecCreator("Dataset File", StringCell.TYPE).createSpec();
        DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
        
        return new DataTableSpec[]{outputSpec, outputSpec};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {

        // TODO save user settings to the config object.
        
        m_ratio.saveSettingsTo(settings);

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
        
        m_ratio.loadSettingsFrom(settings);

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

        m_ratio.validateSettings(settings);

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

