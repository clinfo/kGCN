package org.gcnk.knime.nodes.GCNVisualizer;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.PrintWriter;

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
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.gcnk.knime.nodes.Activator;
import org.gcnk.knime.nodes.GCNNodeModel;

/**
 * This is the model implementation of GCNVisualizer.
 * Generates data for displaying the contribution of each compound of the testset to the predicted value
 *
 * @author org.gcnk
 */
public class GCNVisualizerNodeModel extends GCNNodeModel {
    
    // the logger instance
    private static final NodeLogger logger = NodeLogger
            .getLogger(GCNVisualizerNodeModel.class);
        
    /** the settings key which is used to retrieve and 
        store the settings (from the dialog or from a settings file)    
       (package visibility to be usable from the dialog). */
	static final String CFGKEY_MODEL_PY             = "model.py";
	static final String CFGKEY_WITH_FEATURE         = "With Feature";
	static final String CFGKEY_WITH_NODE_EMBEDDING  = "With Node Embedding";
	static final String CFGKEY_EMBEDDING_DIM        = "Embedding Dim";
	static final String CFGKEY_NORMALIZE_ADJ_FLAG   = "Normalize Adj Flag";
	static final String CFGKEY_SPLIT_ADJ_FLAG       = "Split Adj Flag";
	static final String CFGKEY_ORDER                = "Order";

	static final String  DEFAULT_MODEL_PY             = "sample_chem.singletask.solubility.model";
	static final Boolean DEFAULT_WITH_FEATURE         = true;
	static final Boolean DEFAULT_WITH_NODE_EMBEDDING  = false;
	static final Integer DEFAULT_EMBEDDING_DIM        = 10;
	static final Boolean DEFAULT_NORMALIZE_ADJ_FLAG   = false;
	static final Boolean DEFAULT_SPLIT_ADJ_FLAG       = false;
	static final Integer DEFAULT_ORDER                = 1;


    // example value: the models count variable filled from the dialog 
    // and used in the models execution method. The default components of the
    // dialog work with "SettingsModels".
    private final SettingsModelString m_model_py =
    		new SettingsModelString(
    				CFGKEY_MODEL_PY,
    				DEFAULT_MODEL_PY);

    private final SettingsModelBoolean m_with_feature =
    		new SettingsModelBoolean(
    				CFGKEY_WITH_FEATURE,
    				DEFAULT_WITH_FEATURE);

    private final SettingsModelBoolean m_with_node_embedding =
    		new SettingsModelBoolean(
    				CFGKEY_WITH_NODE_EMBEDDING,
    				DEFAULT_WITH_NODE_EMBEDDING);

    private final SettingsModelIntegerBounded m_embedding_dim =
    		new SettingsModelIntegerBounded(
                    CFGKEY_EMBEDDING_DIM,
                    DEFAULT_EMBEDDING_DIM,
                    1, Integer.MAX_VALUE);

    private final SettingsModelBoolean m_normalize_adj_flag =
    		new SettingsModelBoolean(
    				CFGKEY_NORMALIZE_ADJ_FLAG,
    				DEFAULT_NORMALIZE_ADJ_FLAG);

    private final SettingsModelBoolean m_split_adj_flag =
    		new SettingsModelBoolean(
    				CFGKEY_SPLIT_ADJ_FLAG,
    				DEFAULT_SPLIT_ADJ_FLAG);

    private final SettingsModelIntegerBounded m_order =
    		new SettingsModelIntegerBounded(
                    CFGKEY_ORDER,
                    DEFAULT_ORDER,
                    1, Integer.MAX_VALUE);

    /**
     * Constructor for the node model.
     */
    protected GCNVisualizerNodeModel() {
    
        // TODO one incoming port and one outgoing port is assumed
        super(2, 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected BufferedDataTable[] execute(final BufferedDataTable[] inData,
            final ExecutionContext exec) throws Exception {

    	// Get DatasetFile    	
    	String modelFile        = getInPortFileNoCheck("Model File", inData[0]);
    	String datasetFile      = getInPortFile       ("Dataset File", inData[1]);
        String datasetCleanFile = datasetFile.substring(0, datasetFile.lastIndexOf('.')) + "_clean.jbl";
        String workDir          = new File(datasetFile).getParent();

        String pythonPath = System.getenv("GCNK_PYTHON_PATH");
        
        String logFile    = workDir + "/GCNVisualizer.log";
        String outDir     = workDir + "/visualization";
        String configFile = workDir + "/visualize.json";

        // Clear dataset file
        {
	        String scriptPath = Activator.getFile("org.gcnk.knime.nodes", "/py/clean_dataset.py").getAbsolutePath();
	        String[] Command = { pythonPath, scriptPath, "--dataset", datasetFile, "--output", datasetCleanFile };
	        
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
        }
        
        // json configuration file.
        String model_py             = String.valueOf(m_model_py            .getStringValue());
    	String with_feature         = String.valueOf(m_with_feature        .getBooleanValue());
        String with_node_embedding  = String.valueOf(m_with_node_embedding .getBooleanValue());
        String embedding_dim        = String.valueOf(m_embedding_dim       .getIntValue());
    	String normalize_adj_flag   = String.valueOf(m_normalize_adj_flag  .getBooleanValue());
    	String split_adj_flag       = String.valueOf(m_split_adj_flag      .getBooleanValue());
        String order                = String.valueOf(m_order               .getIntValue());
        
        String model   = modelFile       .replaceAll("\\\\", "/");
        String dataset = datasetCleanFile.replaceAll("\\\\", "/");

        PrintWriter pw = new PrintWriter(configFile);
        pw.println("{");
        pw.println("    \"model.py\"            : \"" + model_py             + "\",");
        pw.println("    \"load_model\"          : \"" + model                + "\",");
        pw.println("    \"dataset\"             : \"" + dataset              + "\",");
        pw.println("    \"with_feature\"        : "   + with_feature         + ",");
        pw.println("    \"with_node_embedding\" : "   + with_node_embedding  + ",");
        if( m_with_node_embedding.getBooleanValue() )
	    	pw.println("    \"embedding_dim\"       : " + embedding_dim      + ",");
        pw.println("    \"normalize_adj_flag\"  : "   + normalize_adj_flag   + ",");
        pw.println("    \"split_adj_flag\"      : "   + split_adj_flag       + ",");
        pw.println("    \"order\"               : "   + order                );
        pw.println("}");
        pw.close();

        new File(outDir).mkdir();

        // Visualize
        {
	        String scriptPath = System.getenv("GCNK_SOURCE_PATH") + "/gcn.py";
	
	        String[] Command = { pythonPath, scriptPath, "visualize", "--config", configFile};
	        
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
        }
        
        // the data table spec of the single output table, 
        // the table will have three columns:
        DataColumnSpec[] allColSpecs = new DataColumnSpec[2];
        allColSpecs[0] = new DataColumnSpecCreator("Log File"               , StringCell.TYPE).createSpec();
        allColSpecs[1] = new DataColumnSpecCreator("Visualization Directory", StringCell.TYPE).createSpec();
        DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
        // the execution context will provide us with storage capacity, in this
        // case a data container to which we will add rows sequentially
        // Note, this container can also handle arbitrary big data tables, it
        // will buffer to disc if necessary.
        BufferedDataContainer container = exec.createDataContainer(outputSpec);
        {
	        RowKey key = new RowKey("Files");
	        DataCell[] cells = new DataCell[2];
	        cells[0] = new StringCell(logFile);
	        cells[1] = new StringCell(outDir);
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
    	CheckGCNKSourcePath();
    	
        // check spec with selected column
    	CheckInportFile("Model File"  , inSpecs[0]);
    	CheckInportFile("Dataset File", inSpecs[1]);

        DataColumnSpec[] allColSpecs = new DataColumnSpec[2];
        allColSpecs[0] = new DataColumnSpecCreator("Log File"               , StringCell.TYPE).createSpec();
        allColSpecs[1] = new DataColumnSpecCreator("Visualization Directory", StringCell.TYPE).createSpec();
        DataTableSpec outputSpec = new DataTableSpec(allColSpecs);
        
        return new DataTableSpec[]{outputSpec};
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {

        // TODO save user settings to the config object.
    	
    	m_model_py            .saveSettingsTo(settings);
        m_with_feature        .saveSettingsTo(settings);
        m_with_node_embedding .saveSettingsTo(settings);
        m_embedding_dim       .saveSettingsTo(settings);
        m_normalize_adj_flag  .saveSettingsTo(settings);
        m_split_adj_flag      .saveSettingsTo(settings);
        m_order               .saveSettingsTo(settings);

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
        
    	m_model_py            .loadSettingsFrom(settings);
        m_with_feature        .loadSettingsFrom(settings);
        m_with_node_embedding .loadSettingsFrom(settings);
        m_embedding_dim       .loadSettingsFrom(settings);
        m_normalize_adj_flag  .loadSettingsFrom(settings);
        m_split_adj_flag      .loadSettingsFrom(settings);
        m_order               .loadSettingsFrom(settings);

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

    	m_model_py            .validateSettings(settings);
        m_with_feature        .validateSettings(settings);
        m_with_node_embedding .validateSettings(settings);
        m_embedding_dim       .validateSettings(settings);
        m_normalize_adj_flag  .validateSettings(settings);
        m_split_adj_flag      .validateSettings(settings);
        m_order               .validateSettings(settings);

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

