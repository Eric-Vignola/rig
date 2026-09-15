"""
Generic Maya constants.
"""

import enum


class Renderers:
    """Default renderers available in Maya"""

    SOFTWARE = "mayaSoftware"
    HARDWARE = "mayaHardware2"
    VECTOR   = "mayaVector"
    ARNOLD   = "arnold"


class Axis(enum.Enum):
    X = 0
    Y = 1
    Z = 2


class ImageFormats(enum.Enum):
    """The list of formats supported by the internal maya renderers (software, hardware and vector).

    While most are interchangeable between renderers, not all are equally supported and some may not
    be available based on the operating system that is being used.
    """

    def __str__(self):
        """Overrides built in method.

        Uses the value of the enum when this object is used in string operations.
        """
        return self.extension

    @property
    def format_name(self):
        """Gets the format name maya expects."""
        return self.value

    # TODO: Cache this value
    @property
    def extension(self):
        """Gets the extension associated with the enunm."""
        return self.name.split("_", 1)[0].lower()

    GIF         = "GIF"
    PIC         = "SoftImage"
    RLA         = "RLA"
    TIFF        = "Tiff"
    TIFF_16     = "Tiff16"
    SGI         = "SGI"
    PIX         = "Alias PIX"
    IFF         = "Maya IFF"
    JPG         = "JPEG"
    EPS         = "EPS"
    IFF_16      = "Maya16 IFF"
    YUV         = "Quantel"
    SGI_16      = "SGI16"
    TGA         = "Targa"
    BMP         = "Windows Bitmap"
    SGI_MOVIE   = "SGI Movie"
    MOV         = "Quicktime"
    AVI         = "AVI"
    PNT         = "MacPaint"
    PSD         = "PSD"
    PNG         = "PNG"
    PICT        = "QuickDraw"
    QIF         = "QuickTime Image"
    DDS         = "DDS"
    PSD_LAYERED = "PSD Layered"
    EXR         = "EXR(exr)"
    IMF         = "IMF plugin"
    SWF         = "Macromedia SWF (swf)"
    AI          = "Adobe Illustrator (ai)"
    SVG         = "SVG (svg)"
    SWFT        = "Swift3DImporter (swft)"

    # This is not an extension but a value to indicate that you want to enable a custom extension instead
    CUSTOM = "Custom Image Format"


class EvaluationManagerModes:
    """The different modes the Evaluation Manager set to."""

    OFF      = "off"
    SERIAL   = "serial"
    PARALLEL = "paralell"


class HardwareRenderingModes:
    """The different hardware rendering modes."""

    WIRE                        = 0
    SHADED                      = 1
    WIRE_ON_SHADED              = 2
    DEFAULT_MATERIAL            = 3
    SHADED_AND_TEXTURED         = 4
    WIRE_ON_SHADED_AND_TEXTURED = 5
    BOUNDING_BOX                = 6